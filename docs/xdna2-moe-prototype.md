# XDNA2 MoE expert prototype

This branch contains an opt-in bring-up path for running selected
DeepSeek-V4-Flash routed experts on Strix Halo's 32-tile XDNA2 NPU. HIP keeps
attention, routing, shared experts, and residual combination; CPU code handles
control and the current SwiGLU boundary. It is a measurement vehicle, not a
production acceleration claim, and the normal HIP image is unchanged.

The design follows AMD's GPT-OSS-20B QMoE split—host routing followed by
selected-expert accelerator work—and adapts TileFuse's full 4x8 XDNA2 GEMV
dataflow to decode Ember's byte-exact affine ROCMFP2 blocks. Provenance and
pins are recorded in `providers/xdna2/VENDOR.md`.

## Lessons carried forward from ViT-Scout

The design explicitly incorporates the negative and positive results in
[Ryzen AI / Strix Halo NPU — Findings](https://git.otheru.ai/otheru/vit-scout/src/branch/main/RYZEN_AI_FINDINGS.md):

- **Correctness gates every benchmark.** A compiled xclbin, low latency, or an
  active-NPU signal is not evidence of correct output. The legacy ViT result
  timed silently uncorrelated multi-output data. Ember must compare each NPU
  projection to the ROCMFP2 host reference, then run the full differential
  validator before reporting throughput.
- **Dispatch-heavy hybrid paths usually lose.** ViT's small custom kernels
  reached only 0.07% of peak, and AMD's hybrid LLM artifacts were 8x slower
  than pure NPU in one measured case. This prototype therefore uses all 32
  compute tiles, persistent instruction/context objects, bounded persistent
  weight BOs, and selective BO synchronization. Its current three launches per
  selected expert are still a known red flag; a fused expert runlist is a
  prerequisite for promotion.
- **Use realistic data and end-to-end timing.** Random weights inflated one CPU
  baseline by roughly 14x. Ember's decision data must use the published model,
  include first-use packing/cache misses and GPU/NPU synchronization, and
  report steady state separately.
- **Do not infer a small fixed NPU memory ceiling from one `ENOSPC`.** A later
  13.1 GiB-RSS pure-NPU model worked. Ember exposes a bounded cache and records
  eviction/miss behavior rather than baking in the earlier 3 GiB assumption.
- **UMA is not an allocation-coherence API.** ROCm and XRT still require their
  own ownership and synchronization. Model bytes remain mmap-backed on the
  host; packed XRT BOs are synchronized once on cache fill, then only the small
  activations/results move per invocation.
- **Telemetry is limited.** On the tested Linux stack, `xrt-smi` exposed an
  active partition but no per-NPU utilization or power. Benchmarks must report
  package power as package power, never relabel it as NPU power.

The report's successful SDK recipe—BF16 transformer graphs through VitisAI—is
also important, but it does not directly consume the model's ROCMFP2 expert
weights. A future whole-expert BF16/VitisAI experiment is worth comparing; it
does not replace the byte-exact custom-kernel correctness baseline.

## Implemented pieces

- `MoeExpertCompute` can decline shapes and safely replay on the existing cold
  path. `DFLASH_MOE_XDNA_REQUIRED=1` turns failures fatal for validation.
- The versioned provider ABI passes global expert IDs, routing weights, and
  ROCMFP2 weight views whose local/global indexing is explicit. The plugin does
  not parse or remap the 85.3 GiB GGUF; hybrid placement exposes its validated
  mmap regions directly.
- `rocmfp2_pack.cpp` losslessly permutes GGML output-major blocks into 128x64
  AIE tiles. Its raw and packed F32 references are tested for exact equality.
- `kernel/rocmfp2_gemv.py` implements the full XDNA2 4x8 object-FIFO topology
  for the exact 4096->2048 and 2048->4096 expert projections.
- `kernel/rocmfp2_gemv.cc` decodes ROCMFP2 directly. Generation 2 keeps the
  K-tile carry in a tile-local F32 buffer and converts to BF16 only at the DMA
  boundary; generation 1 rounded through its BF16 output FIFO after every
  128 inputs.
- `provider_xrt.cpp` owns persistent XRT contexts/instruction BOs and a bounded
  LRU of pre-tiled host-only weight BOs. Generation 2 submits every selected
  gate/up projection in one XRT runlist and every down projection in a second
  runlist, with the CPU SwiGLU calculation as the unavoidable phase boundary.
  It accepts q=1 and separate gate/up/down ROCMFP2 tensors only.
- The optional `release-xdna` image builds XRT plus the XDNA userspace shim from
  the pinned `amd/xdna-driver` tree, builds both AOT kernels with Peano, and
  packages the plugin and artifacts. It never packages a kernel module.

## Build and run the image

The host needs a compatible `amdxdna` kernel driver and firmware, a visible
`/dev/accel/accel0`, and membership in the render group. The container cannot
install or replace these host components. Validate the host first:

```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
```

IOMMU translation is a hard host prerequisite on the tested Strix Halo system.
With `amd_iommu=off`, the accelerator cannot establish the DMA mappings used by
XRT. Remove that kernel argument, reboot, and confirm both `/dev/accel/accel0`
and translated AMD-Vi domains before testing the provider.

Build the opt-in image (the XRT/Peano stages are intentionally much heavier
than the normal release image):

```bash
docker build --target release-xdna -f docker/Dockerfile \
  -t ember:xdna-local .
```

Run the packaged hardware correctness probe before loading model weights:

```bash
docker run --rm --device /dev/accel/accel0 \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  ember:xdna-local ember-xdna-validate
```

The probe enters through the public provider ABI and checks cold and warm-cache
execution against a BF16-aware CPU reference. It is deliberately a structured
bring-up gate, not a substitute for trained-weight differential validation.

Start it with the normal Compose definition plus the XDNA override:

```bash
docker compose -f compose.yaml -f compose.xdna.yaml up --build -d
```

The override passes `/dev/accel/accel0`, sets unlimited memlock, enables the
packaged provider, and leaves enough experts cold to exercise it. The provider
is optional by default so a failure falls back to the baseline path. Use
`DFLASH_MOE_XDNA_REQUIRED=1` only for equivalence testing and benchmarks.

## Controls

| variable | default | purpose |
|---|---:|---|
| `DFLASH_MOE_XDNA_PLUGIN` | unset | Provider `.so`; packaged override supplies it |
| `DFLASH_MOE_XDNA_REQUIRED` | `0` | Fail rather than replay on the baseline path |
| `DFLASH_MOE_XDNA_MIN_TOKENS` | `1` | Smallest batch accepted by the ABI wrapper |
| `DFLASH_MOE_XDNA_TRACE` | `0` | Engine and provider call/cache counters |
| `EMBER_XDNA_ARTIFACT_DIR` | packaged path | xclbin and instruction directory |
| `EMBER_XDNA_DEVICE` | `0` | XRT device index |
| `EMBER_XDNA_WEIGHT_CACHE_MB` | `1024` | Bounded packed-weight BO cache |
| `EMBER_XDNA_KERNEL_GEN` | `2` | Select packaged generation `1` or `2` |
| `EMBER_XDNA_VALIDATION_DENSE` | unset | Use deterministic dense weights in the standalone probe |
| `EMBER_XDNA_VALIDATION_EXPERTS` | `1` | Probe 1-6 selected experts in one call |

`DFLASH_EXPERT_BUDGET_MB` must leave cold experts in the hybrid placement. If
all routed experts are resident on the GPU, the provider has no work.

## Hardware validation status (2026-08-15)

The prototype was exercised on the target gfx1151 Strix Halo host after
enabling IOMMU. XRT identified `RyzenAI-npu5`, AIE2P 6x8, firmware 1.1.2.65.
The stock XRT validation workload reported 51 TOPS GEMM, 50 us latency, and
94,160 operations/s, proving the packaged XRT/driver/device path is functional.

Generation 1's structured `ember-xdna-validate` case passed cold and warm
execution with zero maximum absolute error and cosine similarity 1.0 (about
102 ms cold and 100 ms warm). A deterministic dense A/B exposed its repeated
BF16 carry rounding: maximum absolute error 0.15625, RMS error 0.08283, and
cosine similarity 0.999375.

Generation 2 packages separate AOT artifacts for both projection shapes. Its
tile-local F32 carry retained exact structured output (101.83 ms cold,
100.27 ms warm) and improved the same dense case to maximum absolute error
0.01250, RMS error 0.00424, and cosine similarity 0.9999945. This is a 19.5x
RMS reduction, but the dense probe still deliberately fails the strict
`1e-4` gate. Keeping the output FIFO BF16 matters: an earlier F32 object-FIFO
attempt appeared to run in roughly 54 ms but produced cosine similarity near
0.01 because its host/DMA layout was wrong.

The model-default six-expert runlist also passed the structured reference. Two
calls executed 36 projection kernels through four host submissions, instead of
36 submissions, but warm latency was still 560.25 ms. The six-expert dense
case took 799.73 ms warm and accumulated 0.075 maximum absolute error. XRT
runlists therefore solve submission scaling but do not solve scalar kernel
throughput.

The trained 85.3 GiB DeepSeek-V4-Flash model then ran with the provider marked
required, a 32 GiB hot-expert budget, exact prefill, and runtime top-k 4. This
proved all 43 mmap-backed expert layers can reach XRT without fallback. A
self-comparison reproduced four tokens after in-memory snapshot restore, but a
paired first-pass comparison failed immediately:

```text
HIP:  795, 17038, 1137, 4
XDNA: 1718, 82, 1018, 1718
```

The generation-2 trained-model run used exact prefill, runtime top-k 4, a
16 GiB hot-expert budget, and the provider marked required. It reached every
layer and recorded 1,069 calls / 3,389 cold experts / 10,167 projection kernels
as 2,138 runlist submissions. Provider wall time was 476.4 seconds, or roughly
446 ms per call. The differential validator still reproduced the separate
hybrid snapshot mismatch at token index 2 (`41070` versus `5375`), so it did
not establish trained-model equivalence. Generation 2 is more accurate in the
standalone dense probe and far less dispatch-heavy, but is not faster end to
end. Keep the provider opt-in; do not deploy it for model serving or claim
CPU/GPU/NPU acceleration yet.

### What the NPU should do next

Direct scalar ROCMFP2 decode is the wrong steady-state role. A trial that
constructed BF16 vectors inside the tile and used elementwise vector MACs
compiled, but returned non-finite down-projection output on hardware and was
rejected. More importantly, decoding each 2-bit weight into a vector register
still spends scalar instructions in the inner loop.

The next useful prototype should prepack a bounded hot set into a native AIE2P
matrix layout and use the supported BF16/BFP16 matrix-multiply path. The
[official IRON kernel library](https://xilinx.github.io/mlir-aie/dev/api/kernels/)
already exposes vectorized `mm`/`mv` kernels and an AIE2P option that implements
BF16 MMUL with BFP16. That trades cache memory
and first-use conversion for hardware MAC utilization, so it must report both
cold conversion and warm resident latency. Fuse gate, up, SwiGLU, and down in
one persistent graph if the local-memory budget permits; otherwise the two
runlist phases in generation 2 remain the minimum correct boundary.

Until that matrix-native kernel beats HIP, the NPU should stay off the token
critical path. The safer eventual role is coarse, asynchronous work that can
overlap the GPU—such as a compact speculative drafter or background embedding
pipeline—not fine-grained per-layer assistance. UMA makes the bytes physically
reachable, but XRT and ROCm ownership transitions still make small ping-pong
workloads expensive.

The same session exposed a separate hybrid-path snapshot issue in the HIP
control: its fresh run matched its baseline, but restored decoding diverged at
token index 2. That must be fixed before the hybrid differential validator can
serve as a clean oracle for later XDNA work.

Before treating the path as usable on gfx1151 hardware:

1. Compare both individual projection outputs against the F32 ROCMFP2 host
   reference with signed, trained-model activations; include cache hit/miss
   cases and every output lane.
2. Compare the full selected expert (clamped SwiGLU and router weighting) to the
   HIP baseline. Define explicit BF16 tolerances and fail closed.
3. Run Ember's differential validator with
   `DFLASH_MOE_XDNA_REQUIRED=1`, snapshot restore, disk round-trip, two resident
   sessions, and DSpark when configured.
4. Measure end-to-end tok/s and per-token latency against HIP-only, with cold
   packing separated from warm steady state. Record provider launch/sync time,
   total package power, GPU slowdown while XDNA runs, thermals, cache hit rate,
   and `xrt-smi` partition activity.
5. Replace scalar ROCMFP2 decode with a matrix-native packed kernel and compare
   cold conversion, warm cache residency, and end-to-end performance. The
   generation-2 runlists already prove that submission fusion alone is not
   enough. Keep the feature off by default unless both output and end-to-end
   performance pass.

DeepSeek's HC update remains sequential at q=1 in this hybrid path. Raising a
prefill chunk limit without preserving every token boundary is not correct.
