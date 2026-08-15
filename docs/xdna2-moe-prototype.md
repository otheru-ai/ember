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
- `kernel/rocmfp2_gemv.cc` decodes ROCMFP2 directly. It is currently scalar
  bring-up code; TileFuse-style vectorization comes only after hardware
  equivalence.
- `provider_xrt.cpp` owns persistent XRT contexts/instruction BOs and a bounded
  LRU of pre-tiled host-only weight BOs. It accepts q=1 and separate
  gate/up/down ROCMFP2 tensors only.
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

`DFLASH_EXPERT_BUDGET_MB` must leave cold experts in the hybrid placement. If
all routed experts are resident on the GPU, the provider has no work.

## Hardware validation status (2026-08-14)

The prototype was exercised on the target gfx1151 Strix Halo host after
enabling IOMMU. XRT identified `RyzenAI-npu5`, AIE2P 6x8, firmware 1.1.2.65.
The stock XRT validation workload reported 51 TOPS GEMM, 50 us latency, and
94,160 operations/s, proving the packaged XRT/driver/device path is functional.

The structured `ember-xdna-validate` case passed cold and warm execution with
zero maximum absolute error and cosine similarity 1.0 (101.98 ms cold, 99.20 ms
warm in the final packaged image). A dense deterministic stress case did not
meet the correctness gate: maximum absolute error was about 0.0414, RMS error
about 0.0101, and cosine similarity 0.998465. That failure is material even
though the cosine value looks high.

The trained 85.3 GiB DeepSeek-V4-Flash model then ran with the provider marked
required, a 32 GiB hot-expert budget, exact prefill, and runtime top-k 4. This
proved all 43 mmap-backed expert layers can reach XRT without fallback. A
self-comparison reproduced four tokens after in-memory snapshot restore, but a
paired first-pass comparison failed immediately:

```text
HIP:  795, 17038, 1137, 4
XDNA: 1718, 82, 1018, 1718
```

The bring-up implementation also recorded 1,049 provider calls in 370.6 seconds
(roughly 353 ms per call), before model-load and non-provider work. Its three
serial XRT launches per selected expert are therefore both numerically invalid
for inference and a severe performance regression. Keep the provider opt-in;
do not deploy it for model serving or claim CPU/GPU/NPU acceleration yet.

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
5. Replace the three-launch expert sequence with a fused runlist or prove with
   measurement that its dispatch/synchronization overhead still wins. Keep the
   feature off by default unless both output and end-to-end performance pass.

DeepSeek's HC update remains sequential at q=1 in this hybrid path. Raising a
prefill chunk limit without preserving every token boundary is not correct.
