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

This experiment changes MMQ only. In the Qwen MTP frontiers, dense q=4/q=5
verification can reach it, and q=16 can reach it for both dense and routed
expert projections. q=1 and routed experts at q=4/q=5 use MMVQ and therefore
remain exact-int8 negative dispatch controls. The startup warning records
`activation_prepack=on` or `activation_prepack=off`; real-weight timing evidence
must retain that log so two compiler variants cannot be mistaken for one
another.

`DFLASH_ROCMI4_W4A8_DISPATCH_EVIDENCE=1` enables launch evidence for a
non-timed real-weight differential pass. It records the selected ROCMI4 route
and, from inside the exact gfx1151 runtime guard, every actual W4A8 kernel
launch. The release parser requires dense logical q=4 (which intentionally
uses the cached physical q=5 graph), dense q=5 and q=16, and routed-expert q=16
positive controls. It also requires q=1 dense/routed and q=5 routed-expert MMVQ
negative controls and rejects any W4A8 launch for those widths. This telemetry
is never enabled during clean timing; profiler passes opt into the same runtime
variant separately. A startup warning alone is configuration, not dispatch
evidence, and neither kind of evidence is a performance result.

For the screened ROCm 10 gfx1151 compiler images, prepacking reduces the saved
assembly's VGPR counts from 183/190 to 143/141 for unchecked/checked kernels.
That is a compile-resource observation, not a throughput result. Each block's
source- and launch-derived 27,776-byte dynamic LDS allocation allows at most
four workgroups per gfx1151 128-KiB WGP, an eight-wave-per-SIMD LDS upper bound.
The compiler-reported register-only occupancy of ten waves therefore does not
prove higher workgroup residency.

The instruction and fragment contract is pinned to AMD GPUOpen's
`machine-readable-isa/latest` archive downloaded 2026-08-27 (SHA-256
`82404f1126761b7877595b622afa7e1f311f2f41e89a3abe9aaf8ad045c082e2`),
specifically `amdgpu_isa_rdna3_5.xml` (archive-entry timestamp 2026-08-04;
internal release date 2026-02-20; schema 1.1.1; RDNA 3.5 architecture ID 9).
It defines
`V_WMMA_I32_16X16X16_IU4` as a VOP3P WMMA instruction with 64-bit A/B lane
operands and a 256-bit I32 accumulator, and requires two replicated A/B copies
in wave32. The checked-in, reviewable derivation used by the offline gate is
`rdna3_5_iu4_isa_facts.json`; it pins the archive and XML-entry hashes, VOP3P
opcode 69, operand sizes, and modifier bit fields. Do not substitute the RDNA4
fragment layout on gfx1151.

Provenance and commit pins are recorded in `engine/VENDOR.md`. The storage
contract test is `test/test_rocmi4.c`; the exhaustive mixed-signedness algebra
and fragment-packing oracle is `test/test_rocmi4_q8_i4_exact.cpp`.

After every ROCm or compiler change, emit the gfx1151 assembly with
`GGML_HIP_EXPORT_METRICS=ON`, disassemble the matching object with the same
pinned ROCm toolchain, and run the variant-specific gate. For the register-pack
build:

```bash
/opt/rocm/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1151 \
  build-rocm/engine/ggml/src/ggml-hip/\
mmq-instance-q4_0_rocmi4-hip-amdgcn-amd-amdhsa-gfx1151.o \
  > build-rocm/rocmi4-w4a8.disasm
python3 scripts/check_rocmi4_w4a8_isa.py \
  build-rocm/engine/ggml/src/ggml-hip/\
mmq-instance-q4_0_rocmi4-hip-amdgcn-amd-amdhsa-gfx1151.s \
  --disassembly build-rocm/rocmi4-w4a8.disasm \
  --cmake-cache build-rocm/CMakeCache.txt \
  --variant register
```

Use `--variant prepack` for a build configured with
`GGML_HIP_ROCMI4_W4A8_IU4_PREPACK=ON`. The gate binds the requested variant to
`CMakeCache.txt`, requires both boundary-check variants, checks the assembled
VOP3P opcode and modifier fields against the official derivation, proves the
shifted-high dependency order, requires `mmq_x=32`, rejects private segments or
scratch, and applies the screened per-variant VGPR/SGPR/occupancy bounds.
Passing the source-level oracle without this production assembly-and-object
gate is not sufficient to make the experiment eligible for A/B timing. Neither
gate is device execution or a performance result.

The container build exposes the same opt-in controls as
`EMBER_ROCMI4_W4A8_IU4`, `EMBER_ROCMI4_W4A8_IU4_PREPACK`, and
`EMBER_HIP_EXPORT_METRICS` build arguments. A qualifying register-pack or
prepack release/dev image pair sets W4A8 and export-metrics ON (and prepack as
appropriate); ordinary images keep all three OFF. The real-weight gate reads
the dev image's CMake cache, re-runs the saved-assembly plus encoded-object
gate inside that exact image, and requires its installed executable to be
byte-identical to the release candidate before accepting runtime dispatch
evidence.
