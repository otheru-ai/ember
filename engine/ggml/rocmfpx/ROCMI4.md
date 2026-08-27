# ROCMI4

`Q4_0_ROCMI4` is ROCmFPX's exact signed-int4 storage format. Each 32-value
block is 17 bytes: 16 bytes of two's-complement nibbles followed by one finite
unsigned UE4M3 scale byte. Bytes `qs[0..15]` store elements `0..15` in their
low nibbles and elements `16..31` in their high nibbles. Dequantization is
`signed_nibble * ue4m3(scale)` (4.25 bits per weight).

The ordinary MMVQ and MMQ implementations are exact: they sign-extend the
nibbles and reuse the int8 dot-product path. `GGML_HIP_ROCMI4_W4A4=ON` enables
an experimental gfx1151-only prefill path using
`v_wmma_i32_16x16x16_iu4_w32`. It quantizes activations to signed int4 and is
therefore lossy. The option defaults OFF, logs a warning when active, and
falls back to exact int8 MMQ on every architecture other than exact gfx1151.

Provenance and commit pins are recorded in `engine/VENDOR.md`. The focused
GPU-free contract test is `test/test_rocmi4.c`.
