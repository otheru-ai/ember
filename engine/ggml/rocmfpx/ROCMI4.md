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

An independent, exact W4-by-A8 IU4 MMQ experiment can be compiled with
`GGML_HIP_ROCMI4_W4A8_IU4=ON`. It remains inactive unless the process also has
`DFLASH_ROCMI4_W4A8_IU4=1`, and runtime dispatch additionally requires exact
gfx1151. A missing/other environment value, another GPU architecture, or a
build with the CMake option disabled uses the existing exact int8 MMQ path.
The build rejects enabling this experiment together with W4A4 so the lossy
activation quantizer cannot silently change its numerical contract.

The W4A8 path keeps the ordinary signed-q8 activation quantizer and D4 q8_1
scale. For each K32 integer tile it decomposes every activation exactly as
`q8 = low_u4 + 16*high_i4`, evaluates the signed-weight/signed-high product,
multiplies that I32 accumulator by 16, and then accumulates the
signed-weight/unsigned-low product. The completed K32 integer is multiplied by
the same finite UE4M3 weight scale and q8_1 activation scale at the same point
in the float accumulation loop as the int8 control. This changes neither the
GGUF format nor resident weight memory. The experiment is deliberately pinned
to `mmq_x=32`, the only screened ROCm 10.0 width whose checked and unchecked
gfx1151 kernels both compile without scratch or spills; wider schedules are not
eligible A/B inputs. It is off by default and carries no performance claim
until device differential tests and alternating A/B timing pass.

`GGML_HIP_ROCMI4_W4A8_IU4_PREPACK=ON` is a second, compile-time-only
experiment and requires `GGML_HIP_ROCMI4_W4A8_IU4=ON`. During cooperative
global-to-LDS publication it replaces each K32 q8 payload in place with four
signed-high I4 words followed by four unsigned-low I4 words. The four D4 scale
words, LDS footprint, row stride, global bytes, and consumer LDS bytes remain
unchanged. Consumer waves load the two ready-made IU4 fragments instead of
each repeating the register-local nibble packing. The integer and float
accumulation order remains `high_i4*16 + low_u4` at the same scale boundary.
The option defaults OFF, so it changes neither the ordinary exact-int8 path
nor the screened W4A8 control build. It carries no performance claim until it
independently passes the saved-ISA resource gate, device differential
validation, and alternating A/B timing.

The instruction and fragment contract is pinned to AMD GPUOpen's
`machine-readable-isa/latest` archive downloaded 2026-08-27 (SHA-256
`82404f1126761b7877595b622afa7e1f311f2f41e89a3abe9aaf8ad045c082e2`),
specifically `amdgpu_isa_rdna3_5.xml` (archive-entry timestamp 2026-08-04;
internal release date 2026-02-20; schema 1.1.1; RDNA 3.5 architecture ID 9).
It defines
`V_WMMA_I32_16X16X16_IU4` as a VOP3P WMMA instruction with 64-bit A/B lane
operands and a 256-bit I32 accumulator, and requires two replicated A/B copies
in wave32. Do not substitute the RDNA4 fragment layout on gfx1151.

Provenance and commit pins are recorded in `engine/VENDOR.md`. The storage
contract test is `test/test_rocmi4.c`; the exhaustive mixed-signedness algebra
and fragment-packing oracle is `test/test_rocmi4_q8_i4_exact.cpp`.

After every ROCm or compiler change, emit the gfx1151 assembly with
`GGML_HIP_EXPORT_METRICS=ON` and run:

```bash
python3 scripts/check_rocmi4_w4a8_isa.py \
  build-rocm/engine/ggml/src/ggml-hip/\
mmq-instance-q4_0_rocmi4-hip-amdgcn-amd-amdhsa-gfx1151.s
```

The gate requires both boundary-check variants, the official RDNA 3.5 signed
and unsigned IU4 operand modifiers, the shifted-high dependency order,
`mmq_x=32`, no private segment or scratch instructions, and the screened
VGPR/SGPR ceilings. Passing the source-level oracle without this production
assembly gate is not sufficient to make the experiment eligible for A/B
timing.
