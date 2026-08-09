#!/usr/bin/env python3
"""Go/no-go: can XGrammar build a TokenizerInfo from EMBER's vocabulary?

Every other question about constrained decoding is downstream of this one. Each
GrammarCompiler binds to a TokenizerInfo, and that must describe ember's exact
vocabulary -- the byte-exact joyai-llm tokenizer baked into the GGUF -- or the
token masks are wrong in ways that are hard to detect and would corrupt output.

Reads tokenizer.ggml.tokens straight from the GGUF metadata (cheap: it sits near
the head of the file, no tensor data touched), builds a TokenizerInfo, compiles
a DSML grammar, and checks the mask actually constrains.

The grammar is the real target shape: after <invoke name="terminal"> the ONLY
legal continuation must be a <parameter> tag, which is precisely what production
violates.
"""
import struct
import sys

GGUF_MAGIC = b"GGUF"
(U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64) = range(13)
FIXED = {U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
         U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
         U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8)}


class R:
    def __init__(self, f):
        self.f = f

    def n(self, t):
        fmt, size = FIXED[t]
        return struct.unpack(fmt, self.f.read(size))[0]

    def s(self):
        ln = self.n(U64)
        return self.f.read(ln).decode("utf-8", errors="replace")

    def value(self, t, want):
        """Read a value; only materialise it when we actually want it."""
        if t == STR:
            return self.s()
        if t == ARR:
            et = self.n(U32)
            cnt = self.n(U64)
            if not want:
                for _ in range(cnt):
                    if et == STR:
                        self.f.seek(self.n(U64), 1)
                    elif et == ARR:
                        raise RuntimeError("nested array")
                    else:
                        self.f.seek(FIXED[et][1], 1)
                return None
            return [self.value(et, True) for _ in range(cnt)]
        return self.n(t)


def read_vocab(path):
    with open(path, "rb") as f:
        r = R(f)
        assert f.read(4) == GGUF_MAGIC, "not a GGUF"
        ver = r.n(U32)
        r.n(U64)                      # tensor count
        nkv = r.n(U64)
        toks = None
        for _ in range(nkv):
            key = r.s()
            t = r.n(U32)
            want = key == "tokenizer.ggml.tokens"
            v = r.value(t, want)
            if want:
                toks = v
                break                 # everything after is irrelevant here
        return ver, toks


if len(sys.argv) != 2:
    sys.exit("usage: probe_xgrammar_tokenizer.py MODEL.gguf")
MODEL = sys.argv[1]

ver, vocab = read_vocab(MODEL)
print("GGUF v%d, vocab size = %s" % (ver, len(vocab) if vocab else "NOT FOUND"))
if not vocab:
    sys.exit("no tokenizer.ggml.tokens in metadata")

dsml = [t for t in vocab if "DSML" in t]
print("DSML-bearing tokens: %d -> %s" % (len(dsml), dsml[:6]))

import xgrammar as xgr

# ── TokenizerInfo from ember's own vocabulary ────────────────────────────
# GGUF stores tokens with U+2581 for space (sentencepiece style); XGrammar's
# BYTE_FALLBACK vocab type understands that convention.
try:
    ti = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.BYTE_FALLBACK,
                           vocab_size=len(vocab))
    print("TokenizerInfo: OK  (vocab_size=%d, type=BYTE_FALLBACK)" % ti.vocab_size)
except Exception as e:
    print("TokenizerInfo FAILED:", type(e).__name__, e)
    sys.exit(1)

# ── A DSML grammar that makes a parameterless invoke unrepresentable ─────
PIPE = "｜"
EBNF = r'''
root        ::= callblock
callblock   ::= "<PIPEDSMLPIPEtool_calls>" invoke+ "</PIPEDSMLPIPEtool_calls>"
invoke      ::= "<PIPEDSMLPIPEinvoke name=\"" name "\">" param+ "</PIPEDSMLPIPEinvoke>"
param       ::= "<PIPEDSMLPIPEparameter name=\"" name "\" string=\"true\">" text "</PIPEDSMLPIPEparameter>"
name        ::= [a-zA-Z_] [a-zA-Z0-9_]*
text        ::= [^<]*
'''.replace("PIPE", PIPE)

try:
    g = xgr.Grammar.from_ebnf(EBNF)
    print("Grammar.from_ebnf: OK")
except Exception as e:
    print("Grammar FAILED:", type(e).__name__, e)
    sys.exit(1)

try:
    compiler = xgr.GrammarCompiler(ti)
    cg = compiler.compile_grammar(g)
    print("compile_grammar: OK")
except Exception as e:
    print("compile FAILED:", type(e).__name__, e)
    sys.exit(1)

# ── Does the mask actually constrain? ────────────────────────────────────
import torch
matcher = xgr.GrammarMatcher(cg)
mask = xgr.allocate_token_bitmask(1, ti.vocab_size)
matcher.fill_next_token_bitmask(mask)
allowed = int(torch.tensor(mask).view(torch.int32).bitwise_and(
    torch.arange(32).pow(0)).sum()) if False else None
# Count allowed tokens by expanding the bitmask.
bits = torch.tensor(mask, dtype=torch.int32)
n_allowed = 0
for w in bits.flatten().tolist():
    n_allowed += bin(w & 0xFFFFFFFF).count("1")
print("mask at start: %d of %d tokens allowed (%.4f%%)"
      % (n_allowed, ti.vocab_size, 100.0 * n_allowed / ti.vocab_size))
if n_allowed >= ti.vocab_size:
    print("RESULT: mask does NOT constrain -- investigate")
else:
    print("RESULT: mask constrains generation")
