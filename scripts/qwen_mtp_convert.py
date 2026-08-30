#!/usr/bin/env python3
"""Convert a third-party Qwen3.8-Flash-Next MTP GGUF into Ember's companion format.

Ember's MTP loader (`engine/dflash/qwen4exp/qwen4exp_mtp_loader.cpp`) requires a
bespoke packaging: architecture `qwen4exp-mtp`, its own `qwen4exp-mtp.*` key
namespace, an `ember.mtp.matrix_quant_contract` key, `shared_main_weights` true,
exactly 32 tensors, and `mtp.*` / `mtp_*` names. Published companions -- e.g.
`agentionai/Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST-GGUF` -- ship architecture
`qwen4exp` with `blk.<n_main>.*` names and their own embedding and head.

Nothing is requantized. Every type in that companion (F32, Q8_0, Q6_K and
ROCmFP4-FAST) is already in Ember's allow-list, so this is a pure renaming,
splitting, concatenating and metadata rewrite over quantized bytes.

Two reshapes and why they are byte-safe:

  eh_proj [2*n_embd, n_embd] -> mtp_fc_emb + mtp_fc_hc, split along ne0.
      The reference concatenation is
          eh_proj(cat([enorm(inputs_embeds), hnorm(previous_hidden)], dim=-1))
      (DeepSeek-V3 MTP, as carried by vLLM's deepseek_mtp), so the *embedding*
      half is first along the input axis. GGUF stores [in, out], so ne0 splits
      at n_embd: rows 0..n_embd-1 are fc_emb, n_embd..2*n_embd-1 are fc_hc.
      ne0 is the quantization axis, so the split must land on a block boundary
      -- checked below rather than assumed.

  ffn_gate_exps + ffn_up_exps -> mtp.ffn_gate_up_exps, concatenated along ne1.
      Ember's fused layout is contiguous halves per expert, gate then up
      (test_qwen4exp_frontier.cpp:2104-2105). ne1 is not the quantization
      axis, so this is a byte-level interleave of whole rows.

The fc_emb/fc_hc order is the one thing here that is a convention rather than a
shape constraint. If it is wrong the model still loads and generates; what
collapses is MTP acceptance. Ember's is ~0.767, so a wrong split shows up as
acceptance near zero on one short decode. `--swap-fc` flips it without a
re-download.
"""

import argparse
import struct
import sys

GGUF_MAGIC = b"GGUF"

# (blck_size, type_size) for the types this converter passes through.
TYPE_TRAITS = {
    0:   (1, 4),                    # F32
    1:   (1, 2),                    # F16
    8:   (32, 2 + 32),              # Q8_0
    14:  (256, 256 // 2 + 256 // 4 + 256 // 16 + 2),   # Q6_K
    30:  (1, 2),                    # BF16
    101: (32, 32 // 2 + 1),         # Q4_0_ROCMFP4_FAST
}

# GGUF value type ids
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR = range(10)
T_U64, T_I64, T_F64 = 10, 11, 12
FIXED = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I",
         T_I32: "<i", T_F32: "<f", T_BOOL: "<?", T_U64: "<Q", T_I64: "<q",
         T_F64: "<d"}
FIXED_SIZE = {T_U8: 1, T_I8: 1, T_U16: 2, T_I16: 2, T_U32: 4, T_I32: 4,
              T_F32: 4, T_BOOL: 1, T_U64: 8, T_I64: 8, T_F64: 8}


def row_bytes(type_id, ne0):
    if type_id not in TYPE_TRAITS:
        raise SystemExit(f"unsupported tensor type {type_id}")
    blck, tsz = TYPE_TRAITS[type_id]
    if ne0 % blck:
        raise SystemExit(f"ne0 {ne0} is not a multiple of block size {blck}")
    return ne0 // blck * tsz


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        if self.f.read(4) != GGUF_MAGIC:
            raise SystemExit(f"{path}: not a GGUF file (bad magic). A "
                             "full-size file with a zeroed header is a known "
                             "quantizer failure -- size proves nothing.")
        self.version, self.n_tensors, self.n_kv = struct.unpack("<IQQ",
                                                                self.f.read(20))
        self.kv = {}
        for _ in range(self.n_kv):
            key = self.rd_str()
            self.kv[key] = self.rd_typed()
        self.tensors = []
        for _ in range(self.n_tensors):
            name = self.rd_str()
            (n_dims,) = struct.unpack("<I", self.f.read(4))
            dims = list(struct.unpack("<" + "Q" * n_dims, self.f.read(8 * n_dims)))
            (ttype,) = struct.unpack("<I", self.f.read(4))
            (offset,) = struct.unpack("<Q", self.f.read(8))
            self.tensors.append({"name": name, "dims": dims, "type": ttype,
                                 "offset": offset})
        self.alignment = self.kv.get("general.alignment", (T_U32, 32))[1]
        pos = self.f.tell()
        self.data_start = (pos + self.alignment - 1) // self.alignment * self.alignment

    def rd_str(self):
        (n,) = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8")

    def rd_typed(self):
        (t,) = struct.unpack("<I", self.f.read(4))
        return (t, self.rd_value(t))

    def rd_value(self, t):
        if t == T_STR:
            return self.rd_str()
        if t == T_ARR:
            (et,) = struct.unpack("<I", self.f.read(4))
            (n,) = struct.unpack("<Q", self.f.read(8))
            return (et, [self.rd_value(et) for _ in range(n)])
        return struct.unpack(FIXED[t], self.f.read(FIXED_SIZE[t]))[0]

    def read_tensor(self, tensor):
        nbytes = row_bytes(tensor["type"], tensor["dims"][0])
        for d in tensor["dims"][1:]:
            nbytes *= d
        self.f.seek(self.data_start + tensor["offset"])
        blob = self.f.read(nbytes)
        if len(blob) != nbytes:
            raise SystemExit(f"{tensor['name']}: short read, file truncated")
        return blob


def wr_str(out, s):
    b = s.encode("utf-8")
    out.write(struct.pack("<Q", len(b)))
    out.write(b)


def wr_value(out, t, v):
    if t == T_STR:
        wr_str(out, v)
        return
    if t == T_ARR:
        et, items = v
        out.write(struct.pack("<I", et))
        out.write(struct.pack("<Q", len(items)))
        for item in items:
            wr_value(out, et, item)
        return
    out.write(struct.pack(FIXED[t], v))


def split_ne0(blob, type_id, ne0, ne1, at):
    """Split a [ne0, ne1] tensor along ne0, block-aligned."""
    blck, _ = TYPE_TRAITS[type_id]
    if at % blck:
        raise SystemExit(f"ne0 split at {at} is not on a {blck}-element block "
                         "boundary; refusing to cut a quantization block")
    stride = row_bytes(type_id, ne0)
    head_len = row_bytes(type_id, at)
    head, tail = bytearray(), bytearray()
    for row in range(ne1):
        base = row * stride
        head += blob[base:base + head_len]
        tail += blob[base + head_len:base + stride]
    return bytes(head), bytes(tail)


def interleave_experts(gate, up, type_id, ne0, ne1, n_experts):
    """gate,up [ne0, ne1, E] -> [ne0, 2*ne1, E], gate rows then up rows."""
    plane = row_bytes(type_id, ne0) * ne1
    out = bytearray()
    for e in range(n_experts):
        out += gate[e * plane:(e + 1) * plane]
        out += up[e * plane:(e + 1) * plane]
    return bytes(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--source-revision",
                    default="f5d08274bafd880402bd16f5e3e6c514136ec06c",
                    help="value Ember's loader pins; override only with cause")
    ap.add_argument("--swap-fc", action="store_true",
                    help="emit fc_hc from the first eh_proj half instead of the "
                         "second, if MTP acceptance says the convention is "
                         "reversed for this checkpoint")
    args = ap.parse_args()

    src = Reader(args.source)
    arch = src.kv.get("general.architecture", (T_STR, "?"))[1]
    if arch not in ("qwen4exp", "qwen4exp-mtp"):
        raise SystemExit(f"unexpected architecture {arch!r}")
    if arch == "qwen4exp-mtp":
        raise SystemExit("source is already in Ember companion format")

    by_name = {t["name"]: t for t in src.tensors}
    blk = sorted({int(n.split(".")[1]) for n in by_name
                  if n.startswith("blk.")})
    if len(blk) != 1:
        raise SystemExit(f"expected exactly one MTP block, found layers {blk}")
    layer = blk[0]
    pfx = f"blk.{layer}."

    n_embd = src.kv.get("qwen4exp.embedding_length", (T_U32, 2560))[1]
    n_experts = src.kv.get("qwen4exp.expert_count", (T_U32, 512))[1]

    def take(name):
        t = by_name.get(name)
        if t is None:
            raise SystemExit(f"source is missing {name}")
        return t, src.read_tensor(t)

    outputs = []   # (name, dims, type, blob)

    # Straight renames: blk.<layer>.X -> mtp.X
    direct = ["attn_q.weight", "attn_k.weight", "attn_v.weight",
              "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
              "indexer.q_proj.weight", "indexer.k_proj.weight",
              "indexer.q_norm.weight", "indexer.k_norm.weight",
              "ffn_down_exps.weight", "ffn_gate_inp.weight",
              "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight",
              "ffn_up_shexp.weight", "ffn_down_shexp.weight",
              "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight",
              "hc_attn_inject.weight", "hc_ffn_norm.weight",
              "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight"]
    for suffix in direct:
        t, blob = take(pfx + suffix)
        outputs.append(("mtp." + suffix, t["dims"], t["type"], blob))

    # The block-level HC mixer Ember names without the `mtp.` dot prefix.
    for src_name, dst_name in (("output_hc_norm.weight", "mtp_hc_norm.weight"),
                               ("output_hc_down.weight", "mtp_hc_down.weight"),
                               ("output_hc_up.weight", "mtp_hc_up.weight")):
        t, blob = take(src_name)
        outputs.append((dst_name, t["dims"], t["type"], blob))

    t, blob = take(pfx + "nextn.enorm.weight")
    outputs.append(("mtp_pre_emb_norm.weight", t["dims"], t["type"], blob))
    t, blob = take(pfx + "nextn.hnorm.weight")
    outputs.append(("mtp_pre_hc_norm.weight", t["dims"], t["type"], blob))

    # eh_proj [2*n_embd, n_embd] -> fc_emb (embedding half) + fc_hc.
    t, blob = take(pfx + "nextn.eh_proj.weight")
    if t["dims"][0] != 2 * n_embd:
        raise SystemExit(f"eh_proj ne0 {t['dims'][0]} is not 2*n_embd")
    first, second = split_ne0(blob, t["type"], t["dims"][0], t["dims"][1], n_embd)
    emb_half, hc_half = (second, first) if args.swap_fc else (first, second)
    outputs.append(("mtp_fc_emb.weight", [n_embd, t["dims"][1]], t["type"], emb_half))
    outputs.append(("mtp_fc_hc.weight", [n_embd, t["dims"][1]], t["type"], hc_half))

    # gate/up experts -> one fused tensor, gate rows then up rows per expert.
    gt, gate = take(pfx + "ffn_gate_exps.weight")
    ut, up = take(pfx + "ffn_up_exps.weight")
    if gt["dims"] != ut["dims"] or gt["type"] != ut["type"]:
        raise SystemExit("gate and up expert tensors disagree in shape or type")
    fused = interleave_experts(gate, up, gt["type"], gt["dims"][0],
                               gt["dims"][1], gt["dims"][2])
    outputs.append(("mtp.ffn_gate_up_exps.weight",
                    [gt["dims"][0], 2 * gt["dims"][1], gt["dims"][2]],
                    gt["type"], fused))

    # token_embd and output are deliberately dropped: Ember's companion shares
    # the target model's embedding and head, and the loader requires that.

    if len(outputs) != 32:
        raise SystemExit(f"produced {len(outputs)} tensors, Ember requires "
                         "exactly 32 -- refusing to write a file that will "
                         "fail its contract check")

    kv = [
        ("general.architecture", T_STR, "qwen4exp-mtp"),
        ("general.name", T_STR, "Qwen3.8-Flash-Next MTP (converted)"),
        ("qwen4exp-mtp.source_revision", T_STR, args.source_revision),
        ("qwen4exp-mtp.block_count", T_U32, 1),
        ("qwen4exp-mtp.embedding_length", T_U32, n_embd),
        ("qwen4exp-mtp.hyper_connection_count", T_U32, 4),
        ("qwen4exp-mtp.hyper_connection_low_rank", T_U32, 320),
        ("qwen4exp-mtp.attention.head_count", T_U32, 24),
        ("qwen4exp-mtp.attention.head_count_kv", T_U32, 2),
        ("qwen4exp-mtp.attention.key_length", T_U32, 256),
        ("qwen4exp-mtp.indexer.head_count", T_U32, 4),
        ("qwen4exp-mtp.indexer.key_length", T_U32, 128),
        ("qwen4exp-mtp.indexer.top_k", T_U32, 2048),
        ("qwen4exp-mtp.indexer.compress_ratio", T_U32, 4),
        ("qwen4exp-mtp.expert_count", T_U32, n_experts),
        ("qwen4exp-mtp.expert_used_count", T_U32, 10),
        ("qwen4exp-mtp.shared_main_weights", T_BOOL, True),
        ("ember.mtp.matrix_quant_contract", T_STR, "rocmfp4_fast"),
        ("ember.mtp.converted_from", T_STR, args.source),
        ("general.alignment", T_U32, 32),
    ]

    align = 32
    with open(args.output, "wb") as out:
        out.write(GGUF_MAGIC)
        out.write(struct.pack("<IQQ", 3, len(outputs), len(kv)))
        for key, t, v in kv:
            wr_str(out, key)
            out.write(struct.pack("<I", t))
            wr_value(out, t, v)
        offset = 0
        offsets = []
        for _, dims, ttype, blob in outputs:
            offsets.append(offset)
            offset += (len(blob) + align - 1) // align * align
        for (name, dims, ttype, _), off in zip(outputs, offsets):
            wr_str(out, name)
            out.write(struct.pack("<I", len(dims)))
            out.write(struct.pack("<" + "Q" * len(dims), *dims))
            out.write(struct.pack("<I", ttype))
            out.write(struct.pack("<Q", off))
        pos = out.tell()
        out.write(b"\0" * (((pos + align - 1) // align * align) - pos))
        for _, _, _, blob in outputs:
            out.write(blob)
            pad = ((len(blob) + align - 1) // align * align) - len(blob)
            out.write(b"\0" * pad)

    print(f"wrote {args.output}: {len(outputs)} tensors, "
          f"fc split {'swapped' if args.swap_fc else 'embedding-first'}")
    print("Verify by MTP acceptance, not by loading: a reversed fc_emb/fc_hc "
          "split still loads and generates. Ember's acceptance is ~0.767; "
          "near-zero means rerun with --swap-fc.")


if __name__ == "__main__":
    sys.exit(main())
