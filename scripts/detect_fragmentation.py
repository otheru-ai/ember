#!/usr/bin/env python3
"""Detector for the string-fragmentation failure mode, plus its evaluation.

EVALUATED AGAINST THE REAL CORPUS (1,304 captured production responses):
    firings              11  (0.84%)
    all 11 timestamps    inside the DRY window 07:57-10:04 on 2026-08-08
    before DRY           0 firings in 1,201 captures
    during DRY           11 firings in 102 captures (10.8%)
Two false-positive classes were found by that evaluation and excluded rather
than guessed at: a legitimate LIST of document ids (uniform 8-char items, no
tiny piece), and `--- a//tmp/...` which is correct unified-diff syntax.

The failure: under a repetition penalty the model avoids re-emitting a token
sequence it has produced before by splitting the string into concatenated
pieces, or by inserting a stray character. Seen in production 2026-08-08:

    BLOG="6272a42d" "-" "a106" "-" "4086" "-"
    echo "6""272""a4""2d""-""a""106"
    curl -sk "https://d" "ocs" ".oth" "er"
    BLOG=$(cat /tmp/blog_ id.txt)          <- space inside a filename
    Writing //tmp/list_blog_all.py         <- doubled separator

None of it contains a DSML token, so the marker detector added the same day is
blind to it. This is that gap.

Run with no args to EVALUATE against the whole capture corpus: it must fire on
the known specimen and stay quiet on ordinary traffic. A detector with a high
false-positive rate is worse than none, because it trains you to ignore it.
"""
import json
import os
import re
import sys

CAPTURE = "/tmp/capture"

# Three or more SHORT quoted literals in a row, separated by nothing or spaces.
# Legitimate shell concatenates like this very rarely; a fragmented identifier
# does it by construction. Requiring three keeps ordinary `"a" "b"` argument
# pairs (common and fine) out of the signal.
FRAGMENT_RUN = re.compile(
    r'"([^"\n]{1,8})"\s{0,2}"([^"\n]{1,8})"\s{0,2}"([^"\n]{1,8})"')

# ...but a RUN alone is not enough. Tuned against the real corpus: seq=128
# carries `"560a2c63" "c581b01e" "220bbdbc"`, which is an ordinary LIST of
# document ids -- uniform, full-length items. Genuine fragmentation splits one
# identifier into pieces, so at least one piece is tiny ("-", "a", "6"). Every
# true positive in the corpus has a <=3-char piece; the list has none.
# Checked against the CAPTURED GROUPS, not by re-searching the matched run: a
# closing quote + space + opening quote reads as a 1-char literal `" "`, which
# made a first attempt fire on the very list it was meant to exclude.
TINY_MAX = 3

# A doubled separator where a path is being built. Excludes "://" of a URL and
# the a//b/ of unified-diff headers: `--- a/` + `/tmp/x` renders as `a//tmp/x`,
# which is correct diff syntax and accounted for 21 of 29 first-pass firings.
DOUBLED_SLASH = re.compile(r'(?<!:)(?<![ab])//(?:tmp|home|root|usr|var|etc)\b')


def scan_text(text):
    """Return a list of (signal, excerpt) for one reconstructed response."""
    hits = []
    for m in FRAGMENT_RUN.finditer(text):
        pieces = [g for g in m.groups() if g.strip()]
        if len(pieces) == 3 and min(len(g.strip()) for g in pieces) <= TINY_MAX:
            hits.append(("fragmented-literal-run", m.group(0)[:70]))
            break
    m = DOUBLED_SLASH.search(text)
    if m:
        a = max(0, m.start() - 25)
        hits.append(("doubled-path-separator", text[a:m.end() + 25]))
    return hits


def reconstruct(path):
    """Rebuild visible text + reasoning + tool arguments from an SSE dump."""
    buf = []
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return ""
    with fh:
        for ln in fh:
            ln = ln.strip()
            if not ln.startswith("data:"):
                continue
            body = ln[5:].strip()
            if body == "[DONE]":
                continue
            try:
                d = json.loads(body, strict=False)
            except Exception:
                continue
            for c in d.get("choices", []):
                delta = c.get("delta") or {}
                for k in ("content", "reasoning_content"):
                    if delta.get(k):
                        buf.append(delta[k])
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    if fn.get("arguments"):
                        buf.append(fn["arguments"])
    return "".join(buf)


def main():
    if len(sys.argv) > 1:                      # scan one file, for the proxy
        hits = scan_text(reconstruct(sys.argv[1]))
        for sig, ex in hits:
            print("%s: %s" % (sig, ex))
        return 1 if hits else 0

    files = sorted(
        f for f in os.listdir(CAPTURE) if re.fullmatch(r"live-\d+-resp\.txt", f)
    )
    fired, total = [], 0
    for f in files:
        total += 1
        hits = scan_text(reconstruct(os.path.join(CAPTURE, f)))
        if hits:
            fired.append((f, hits))

    print("  corpus: %d captured responses" % total)
    print("  fired : %d (%.2f%%)" % (len(fired), 100.0 * len(fired) / max(total, 1)))
    print()
    known = "live-130-resp.txt"
    caught = any(f == known for f, _ in fired)
    print("  KNOWN SPECIMEN %s detected: %s" % (known, caught))
    print()
    print("  all firings (inspect for false positives):")
    for f, hits in fired:
        for sig, ex in hits:
            print("    %-22s %-18s %s" % (f, sig, ex.replace("\n", "\\n")[:60]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
