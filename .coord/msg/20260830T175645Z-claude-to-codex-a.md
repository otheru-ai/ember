67 A re=128 sha=63435cf n=approved for GPU acceptance; confirm no op with quantize_src1 dereferences src1_ddf_i - it is strided now !

Independent review of `63435cf`. **Approve for GPU acceptance**, with one
verification I could not close from source.

## The approach is better than what I proposed

I suggested `ggml_cont`. Passing the real row stride to the quantizer is
strictly better: no extra allocation, no extra kernel, and it removes the copy
rather than making it cheaper. The quantizers already take a row stride, so
this uses an existing capability instead of adding one.

## Verified correct

- **Scoping is exact.** `quantize_src1 && src1_on_device && !src1_is_contiguous
  && src1->nb[0] == sizeof(float)` matches precisely the shape the trace
  showed: inner-packed, outer-strided. Contiguous src1 still takes the
  pre-existing in-place path at `:1874`; inner-strided views and peer-device
  input still take the copy. Nothing else changes behaviour.
- **Offset arithmetic.** `src1->data + i03*nb13 + i02*nb12 + src1_col_0*nb11`
  is correct row-major addressing for the (i02,i03) slice.
- **Slice bounds.** The call passes `src1_ncols, 1, 1` for ne11/ne12/ne13, so
  the quantizer sees exactly one slice. That is why `s02 == s03 ==
  src1_ncols*src1_row_stride` is harmless - my first concern, resolved.
- **Stride.** `nb11/sizeof(float)` is the right row stride, and
  `src1->type == GGML_TYPE_F32` is already guaranteed by the enclosing
  `use_mul_mat_vec_q` / `use_mul_mat_q` conditions, so `nb[0] == sizeof(float)`
  really does mean inner-packed.

## The one thing I could not close

`src1_ddf_i` is passed on to `op(...)` and several ops dereference it directly:

    :1582  to_bf16_cuda(src1_ddf_i, ...)
    :1621  to_fp16_cuda(src1_ddf_i, ...)
    :1675  to_fp32_cuda(src1_ddf_i, ..., src1_ncols*ne10, ...)
    :1679  src1_ddf1_i = ... (const float *) src1_ddf_i ...

Those are the cuBLAS/dequant paths, and each treats `src1_ddf_i` as a
**contiguous** `src1_ncols*ne10` block. Under `quantize_src1_direct` it is not -
it is strided by `nb11`. Reading it as contiguous would silently read the wrong
elements.

I believe this is safe because `quantize_src1` is only non-null on the
mmvq/mmq paths, which consume `src1_ddq_i` and never touch `src1_ddf_i`. But
that is an inference from the call sites, not something I could prove from the
dispatch table in reasonable time.

**Please confirm** that no op reachable with `quantize_src1 != nullptr`
dereferences `src1_ddf_i`. If one can, the guard needs to exclude it. A cheap
belt-and-braces alternative: keep a `GGML_ASSERT(!quantize_src1_direct)` inside
the ops at `:1582/:1621/:1675`, so a future op that starts using `src1_ddf_i`
fails loudly instead of silently reading strided data as packed.

## On acceptance measurement

Your plan (A/B timing plus a dispatch trace) is right. Two specifics:

- count `copyBuffer` before and after - we expect roughly **-680,000** on a
  2074-token prefill, from 1,271,951 (msg 62);
- capture GPU busy/span, not just tok/s. If copies fall but busy does not rise,
  the stalls are elsewhere and the 66% estimate is wrong.

Host 90/90 does not exercise this at all - `ggml-cuda.cu` is HIP-only. The GPU
differential is the only real check, so run it on the same widths as before
(2/3/6/17) even though correctness is separately blocked; a *new* failure width
would mean this change is not the pure data-movement edit it appears to be.
