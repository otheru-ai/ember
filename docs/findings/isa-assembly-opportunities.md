# ISA review: where assembly (and one non-assembly change) could still pay

Read against the gfx1151 spec mirror at `/srv/isa`, 2026-09-02, focused on the
kernels that run for the shipped recipe: `Q2_0_ROCMFP2` (129 expert tensors, the
largest in the model) and `Q4_0_ROCMFP4_FAST`.

## Already done — do not redo

* **DP4A is in use.** `ggml_cuda_dp4a` maps to `__builtin_amdgcn_sudot4` on
  RDNA3, i.e. `V_DOT4_I32_IU8`. Both quantized vec_dots use it.
* **The 2-bit LUT gather is already one instruction.**
  `rocmfpx_pack4_fp2_bits8_vec_cuda` uses `__builtin_amdgcn_perm`
  (`V_PERM_B32`) for the code-to-value lookup.
* **`V_DOT8_I32_IU4` is already implemented** — behind `GGML_ROCMI4_W4A8_IU4` /
  `DFLASH_ROCMI4_W4A8_IU4` in `mmq.cu`. It serves type 108 `Q4_0_ROCMI4`, which
  the DeepSeek recipe does not use. The instruction exists on gfx1151 (VOP3P, no
  DPP, no VOPD) and the code exists; only the type is unshipped.

## The real opportunity is not assembly — half the dp4a is redundant

`vec_dot_rocmfpx_fp2_q8_1` issues **two** dp4a per iteration:

```c
sumi = ggml_cuda_dp4a(val_packed,  u, sumi);   // weights . activations
sumq = ggml_cuda_dp4a(0x01010101, u, sumq);    // sum of activations
```

Per 32-element block that is 16 dp4a, of which **8 exist only to sum the
activations** for the affine offset term (`value = code*scale - offset`).

`block_q8_1` already carries that sum. `quantize.cu` stores
`ds = make_half2(d, sum)`, and the struct documents `s` as `d * sum(qs[i])`.
Both `iqs` halves of a rocmfp2 block read the *same* `bq8_1` and the *same*
`e[0]`/`e[1]`, so summed over the two calls the offset contribution is
`offset * d * sum(q)` — which is `offset * __high2float(ds)`.

**So the offset term can be applied once per block from an existing field, and
all 8 `sumq` dp4a deleted: a 50% cut in dp4a issue for the largest tensors in
the model.**

Shape of the change: subtract `offset * __high2float(bq8_1->ds)` on the
`iqs == 0` call only, subtract nothing on `iqs == 1`, drop `sumq` entirely.

Two things to settle before trusting it, neither of which I can do off-box:

1. **Numerics.** `ds.y` is the float sum of the original values, while `sumq` is
   the integer sum of the quantised codes. They agree only up to rounding, so
   this is a small numerical change, not a pure refactor. It needs the
   behavioural gate, not just a build.
2. **Call-pairing.** The saving assumes both halves of every block are always
   evaluated and accumulated. If any path evaluates one half alone the offset
   would be wrong. Confirm in the MMVQ driver before landing.

## Tested and rejected

Building the `V_PERM_B32` selector word costs ~11 ALU ops (4 shifts, 4 masks,
3 ORs). A single-multiply bit-spread, `(bits8 * 0x00041041) & 0x03030303`, would
replace all of it. **It does not work**: 96 of 256 byte values disagree, because
the shifted copies carry into one another. Recorded so nobody re-derives it.

## What the ISA says about the remaining ideas

* `V_DOT2_F32_F16` and `V_DOT2_F32_BF16` are VOP3P **and DPP16-capable** — worth
  knowing if any F16 reduction is still scalar.
* Neither `V_DOT4_I32_IU8` nor `V_DOT8_I32_IU4` has a VOPD encoding, so they
  cannot be dual-issued. Consistent with the standing finding that VOPD is
  unreachable from inline asm, and it means dual-issue is not a lever on the
  quantized dot path specifically.


# Addendum: the per-layer causal mask rebuild (2026-09-02)

`build_mla_attention` allocates and zero-fills
`std::vector<float> mvals(n_attn * n_tokens)` and rebuilds the causal mask on
**every call, and it is called once per layer**. The fused path already avoids
this — it pre-builds `fg.mask_bundle` once and hands each layer a view — but the
`layer_major_batch` prefill path does not.

Sizing: at 2048 tokens and n_attn 4096 that is a 32 MB allocate-and-zero per
layer, roughly **1.4 GB of memset per chunk** across 43 layers, on top of the
predicate cost removed in `f4bba5a`.

## My first idea was wrong

I proposed hoisting the mask out of the layer loop as layer-invariant. **It is
not.** `const int ratio = w.compress_ratios[layer_idx]` — the compressor ratio
is per layer, and it feeds `n_comp_live`, hence `n_attn`, hence the mask shape
and contents. A blanket hoist would produce a wrong mask for most layers.

## What is actually available

The mask depends on the layer *only through* `ratio`. The shipped model has
**three distinct ratios across 43 layers**:

    deepseek4.attention.compress_ratios = [0, 0, 4, 128, 4, 128, ...]
    distinct {0, 4, 128}, counts {0: 2, 4: 21, 128: 20}

So the mask can be **memoised by ratio: 43 builds become 3**, provided layers
sharing a ratio also share `lc.n_comp` and `comp_kv->ne[1]`. Both are per-layer
fields, so that is an assumption to verify in the cache, not to assume — it is
the same class of mistake as the hoist above.

`n_prior_rows = min(kv_start, w.n_swa)` and `n_raw` on this path are
layer-independent, so they are not obstacles.

Order of work: verify the shared-ratio assumption first, since getting it wrong
is a correctness bug rather than a slower kernel. Then memoise.
