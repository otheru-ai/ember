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
  AIE tiles. Gen4 additionally creates its vector-friendly uint4/BF16 layout.
  Raw and both packed F32 references are tested for exact equality.
- `kernel/rocmfp2_gemv.py` implements the full XDNA2 4x8 object-FIFO topology
  for the exact 4096->2048 and 2048->4096 expert projections.
- `kernel/rocmfp2_gemv.cc` decodes ROCMFP2 directly. Generation 2 keeps the
  K-tile carry in a tile-local F32 buffer and converts to BF16 at the output
  DMA boundary; generation 1 rounded through its BF16 FIFO after every 128
  inputs. Generation 3 makes the output FIFO and host BO F32, so gate/up and
  down projections cross the CPU phase boundary without another BF16 round.
- `kernel/rocmfp2_gemv_v4.cc` vector-decodes 64 output lanes at once into a
  tile-local BF16 scratchpad, then performs native eight-row vector MACs into
  the Gen3 FP32 boundary.
- `provider_xrt.cpp` owns persistent XRT contexts/instruction BOs and a bounded
  LRU of pre-tiled host-only weight BOs. Generations 2 through 4 submit every
  selected gate/up projection in one XRT runlist and every down projection in
  a second runlist, with the CPU SwiGLU calculation as the unavoidable phase
  boundary. It accepts q=1 and separate gate/up/down ROCMFP2 tensors only.
- The optional `release-xdna` image builds XRT plus the XDNA userspace shim from
  the pinned `amd/xdna-driver` tree, builds every generation for both projection
  shapes with Peano, and packages the plugin and artifacts. It never packages
  a kernel module.

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
| `EMBER_XDNA_KERNEL_GEN` | `4` | Select packaged generation `1`-`6`; Gen5 is opt-in and Gen6 is validator-only |
| `EMBER_XDNA_VALIDATION_WARM_RUNS` | `1` | Repeat cached validation dispatches for sustained-load and overlap measurements |
| `EMBER_XDNA_VALIDATION_DENSE` | unset | Use deterministic dense weights in the standalone probe |
| `EMBER_XDNA_VALIDATION_EXPERTS` | `1` | Probe 1-6 selected experts in one call |

`DFLASH_EXPERT_BUDGET_MB` must leave cold experts in the hybrid placement. If
all routed experts are resident on the GPU, the provider has no work.

## Hardware validation status (2026-08-15 through 2026-08-16)

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

Generation 3 corrects that layout end to end: its output object FIFO, runtime
sequence type, XRT BO allocation, and host decode are all F32. The one-expert
structured case remained bit-exact at 99.17 ms warm. The one-expert dense case
passed the strict gate with maximum absolute error `7.75e-7`, RMS error
`1.43e-7`, and cosine similarity 1.0 at 139.94 ms warm. This is about 16,100x
lower maximum error and 29,800x lower RMS error than generation 2, without a
measurable latency regression. The cost is twice the result-DMA and host-BO
footprint for each projection; at these vector sizes scalar compute still
dominates that extra transfer.

The model-default six-expert runlist also passed the structured reference. Two
calls executed 36 projection kernels through four host submissions, instead of
36 submissions, but warm latency was still 560.25 ms. The six-expert dense
case took 799.73 ms warm and accumulated 0.075 maximum absolute error. XRT
runlists therefore solve submission scaling but do not solve scalar kernel
throughput.

Generation 3 also passed the six-expert structured and dense gates. Dense warm
latency was 799.37 ms, maximum absolute error was `4.29e-6`, RMS error was
`8.74e-7`, and cosine similarity was 1.0. The output-boundary optimization
therefore fixes the standalone accuracy gate, but deliberately does not claim
an inference acceleration: scalar ROCMFP2 decode still dominates latency.

Generation 4 replaces the scalar decoder with the vector GEMV structure from
the pinned TileFuse W4A16 kernel. Cache fill transposes each 128x64 tile to
K-major order, expands two FP2 codes into uint4 nibbles, and converts UE4M3
scale/offset metadata to exact BF16. Each AIE core vector-unpacks and
dequantizes all 64 output lanes into a 16 KiB tile-local scratchpad, then uses
eight-row `aie::accumulate` operations with an FP32 accumulator. The linked hot
function is `0x370` bytes and contains native unpack, vector multiply/subtract,
and vector MAC instructions with none of the scalar `__mulsf3`, `__divsf3`, or
`__floatunsisf` helpers. Packed cache size doubles from 7.5 MiB to 15 MiB per
complete three-projection expert, remaining below BFP16's 27 MiB and BF16's
48 MiB.

On the same Strix Halo hardware, Gen4's one-expert structured case was bit
exact at 6.58 ms warm. Its dense case retained Gen3's maximum absolute error
`7.75e-7`, RMS error `1.43e-7`, and cosine 1.0 at 6.44 ms warm: a 21.7x
improvement over Gen3's 139.94 ms. The six-expert dense runlist passed at
8.32 ms warm with maximum absolute error `4.29e-6`, RMS error `8.74e-7`, and
cosine 1.0. Cold times were 27.11 ms for one expert and 29.15 ms for the
six-slot repeated-weight validator; distinct trained experts must still report
their larger cache-fill cost separately.

Opt-in provider phase timing explains the remaining warm latency. One dense
expert spent 3.27 ms in the combined gate/up runlist and 3.13 ms in the down
runlist; host SwiGLU took 0.005 ms and XRT BO synchronization was below 0.002
ms in each direction. Six experts spent 4.45 ms in gate/up, 3.77 ms in down,
and 0.031 ms in host SwiGLU. The useful next target is therefore the fixed
second device execution boundary. Moving only SwiGLU arithmetic or activation
copies cannot materially improve this path.

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

### Gen5 fused expert

Gen5 implements the next Gen4 optimization without a persistent BF16/BFP16
expert expansion. One fixed spatial graph keeps gate and up results on the
array, applies Ember-compatible clamped SwiGLU, stages the BF16 hidden vector
through NPU DMA, and performs both down-projection passes under one host
dispatch. `EMBER_XDNA_KERNEL_GEN=5` selects it; Gen4 remains the default until
trained-model differential and end-to-end serving gates pass.

AMD's current [IRON operator library](https://github.com/amd/IRON) confirms the
overall shape: its decode SwiGLU sequences gate GEMV, up GEMV, SiLU, multiply,
and down GEMV under one host dispatch. Its full-ELF path is temporal fusion,
however: the command processor reconfigures the array between separate
operators. Ember's measured host copies are already negligible, and published
[*Unlocking the AMD Neural Processing Unit for ML Training on the Client*](https://arxiv.org/html/2504.03083v1)
reports that minimal reconfiguration was 3.5x faster than whole-array
reconfiguration on the first invocation of a new shape. Gen4 should therefore
use one spatial design and one fixed control program, not concatenate five
independent overlays. This is also consistent with the Linux driver model:
firmware executes one `ctrlcode` program which starts host-DDR DMAs and raises
one completion interrupt, as described by the
[upstream `amdxdna` documentation](https://docs.kernel.org/accel/amdxdna/amdnpu.html).

The upstream BF16 SiLU is not accurate enough for Ember. Its own decode test
uses 4% relative and `0.4` absolute tolerances. A 16,384-value hardware sweep
on this Strix Halo found that the unguarded AIE2P tanh approximation can wrap
outside its useful input domain (`tanh(-9.65)` returned `+1`). Explicitly
saturating outside `[-4, 4]` removed that failure, but 2,317 values still
exceeded `1e-4`; maximum absolute tanh error was `0.0371` and RMS error was
`0.00344`. That implementation is rejected. Gen5 instead uses a degree-11,
range-reduced scalar exponential and explicit BF16 round-to-nearest-even. Its
complete scale/clamp/SwiGLU contract matched 16,384 of 16,384 hardware probe
values exactly. A single core processing 256 blocks averaged 94.98 us per
64-value block.

The first full control stream timed out because it waited on an MM2S DMA task
which had not requested a completion token. The supplied
[AIE-RT `release/main_aig` branch](https://github.com/Xilinx/aie-rt/tree/release/main_aig)
documents this contract: a descriptor must issue a task-completion token before
software waits on or reuses it. Explicit `issue_token=True` fixed the replay.
This also avoids a seventh memory-tile DMA channel: hidden and final results use
one sequenced packet channel, exactly filling the six-channel hardware limit.

On physical Strix Halo, the complete one-expert graph measured 3.88 ms cold and
0.544 ms warm with exact structured-weight output. A dense run measured 3.55 ms
cold and 0.513 ms warm, maximum absolute error `3.34e-6`, RMS error `5.01e-7`,
and cosine 1.0. Against Gen4's 6.44 ms dense warm result, the standalone fused
kernel is 12.5x faster. This is a kernel result, not an end-to-end inference
claim; multi-expert runlists, trained weights, cache fill, and HIP comparison
remain required.

The packaged provider validator reproduced the result: one dense expert was
`0.569` ms warm (`26.42` ms cold including fused weight packing), and a
six-entry repeated-weight dense runlist was `2.583` ms warm. Both passed at
cosine 1.0; the six-entry maximum absolute error was `4.29e-6`. The six-entry
kernel path is 3.2x faster than Gen4's `8.32` ms, while retaining Gen4 as the
default.

After fusion, measure trained-model cache-fill cost and routing locality with
the 2x packed representation. Native BFP16 matrix multiplication remains a
fallback experiment if vector GEMV stops scaling, not the immediate next step.
UMA can remove byte copies through dual-registered host pages, but the current
XRT/ROCm stack still requires CPU ownership transitions and fences.

[Ryzen AI 1.8](https://ryzenai.docs.amd.com/en/latest/linux.html) now documents
Strix-class Linux support, while its installer ships an XRT 2.25 package set
and its release notes include Strix Halo in the supported processor family.
This prototype is built and hardware-validated as one pinned XDNA/XRT 2.26
source stack; do not mix its userspace libraries or shim with the independently
versioned installer packages. The host kernel driver and firmware remain host
responsibilities in either installation model.

The same session exposed a separate hybrid-path snapshot issue in the HIP
control: its fresh run matched its baseline, but restored decoding diverged at
token index 2. That must be fixed before the hybrid differential validator can
serve as a clean oracle for later XDNA work.

The packaged Gen5 provider subsequently reached all 43 layers with the trained
85.3 GiB model, a 32 GiB resident-GPU expert budget, exact prefill, and runtime
top-k 4. Its baseline and fresh replay began with the same tokens as the
authoritative monolithic HIP run (`372, 223`); the restored hybrid path alone
changed the second token to `28231`. This localizes the observed differential
failure to the existing hybrid snapshot restore path rather than Gen5 expert
math, although a longer trained-output comparison remains required after that
restore defect is fixed.

End-to-end serving rejects the current target-MoE placement despite the fast
kernel. The exact-prompt cold request measured 26.09 s prefill and 1.58 tok/s
decode; repeating it with packed weights resident improved to 1.74 s prefill
and 10.20 tok/s. The production monolithic fused-GPU/DSpark path on the same
machine measured 671.5 ms prefill and 22.65 tok/s decode. Hybrid placement
disables both the single fused target graph and DSpark, so Gen5's roughly
0.5 ms warm expert execution cannot recover the lost graph-level throughput.
Do not spend the next iteration optimizing cold packing in this target-hybrid
design: even its warm upper bound is less than half the production baseline.

The next boundary investigated was the three-layer DSpark drafter rather than
the 43-layer target. AIE-RT's repeat-count start queue and explicit per-task
completion-token contract made the fixed five-row draft block a plausible way
to amortize host submission without splitting the target graph.

The provider-side prerequisite is implemented and hardware-validated. Gen5
accepts up to five tokens and submits every token/expert pair through one XRT
runlist, keeping input and accumulation surfaces token-private. A dense
five-token/four-expert probe with different activation rows passed at cosine
1.0 and maximum absolute error `9.54e-6`. Its warm provider time was 8.432 ms;
the corresponding one-token/four-expert probe was 1.778 ms, or about 8.89 ms
when repeated five times. Queue amortization therefore saves only about 5%.
Rebatching the same work is insufficient.

The requested draft/head/verify ceiling measurement then rejected drafter
offload. Two identical 32-token production-config DSpark requests were run on
the physical Strix Halo with `DFLASH_DS4_TIMING=1`; the normal service was
restored healthy afterward. The warm request reported 82.2 ms per speculative
step: 5.1 ms draft, 1.7 ms head, and 75.4 ms target verification. End-to-end
decode remained 22.65 tok/s. Even deleting the draft entirely could improve
this step by only about 6%, while measured Gen5 execution for three layers of
five-token/top-k expert work is tens of milliseconds. Do not split the DSpark
drafter around XDNA2: it would replace a 5.1 ms GPU stage with a slower NPU
stage and add three GPU/NPU synchronization points.

An external implementation audit reinforced that decision. The public
[OllamaAMDNPU](https://github.com/BrandedTamarasu-glitch/OllamaAMDNPU) backend
corrected its original 43.7 tok/s NPU-decode claim after discovering that the
benchmark had executed on Vulkan. Its measured XDNA2-only decode reached 1.40
tok/s after a useful 23% kernel-side improvement from balanced
128x128x16 microtiles; weight pre-staging, sync deduplication, larger host
tiles, and runlist batching produced little or no throughput gain. Ember
already uses a 128x64 ROCMFP2 GEMV tile and likewise measured only a 5% gain
from batching. Optimize AIE instruction/dataflow geometry when working on the
kernel, but do not infer an end-to-end win from host submission reduction.

### Direct GPU/XDNA buffer interoperability

The current AMD driver contains DRM PRIME import/export support, and ROCr
exposes `hsa_amd_portable_export_dmabuf`. Ember now packages
`ember-xdna-gpu-interop` to validate the resulting cross-driver contract. The
probe allocates input and output with HIP, exports both as dma-buf handles,
imports them as XRT BOs, runs the existing Gen4 AIE projection, and verifies
the NPU-written result through HIP. It requires both `/dev/kfd`/`/dev/dri` and
`/dev/accel/accel0`, plus unlimited memlock.

On the physical gfx1151 host, ordinary `hipMalloc` passed exactly: the NPU read
the HIP BF16 input and wrote the HIP FP32 output with maximum absolute error
zero. ROCr exported the output as a subrange at offset 8192, which validates
the probe's nonzero-offset XRT sub-buffer handling. Import and initial BO sync
took 0.192 ms once; 20 cached AIE run/wait/output-sync iterations averaged
0.182 ms each for one 4096x2048 projection. This is a real zero-copy byte path,
but the CPU still orders the HIP completion, XRT command, and next HIP work.

`hipMallocManaged` failed export with `HSA_STATUS_ERROR_INVALID_ALLOCATION` on
the same ROCm 7.14/XRT 2.26 stack. Therefore activation and intermediate graph
buffers can use the direct path when they are ordinary HIP allocations, while
the large managed target-weight allocation cannot. NPU weights must remain in
their separately packed XRT cache. The next verifier experiment may use this
zero-copy activation seam, but it must first prove that splitting the 43-layer
fused GPU graph costs less than the expert work moved to XDNA2.

That concurrency gate rejected the per-layer verifier split before invasive
graph surgery. `ember-xdna-validate` supports
`EMBER_XDNA_VALIDATION_WARM_RUNS` so a cached Gen5 workload can remain active
across a real server request. Eight matched 32-token DSpark requests were
measured with and without a concurrent dense five-token/four-expert workload.
The acceptance pattern was identical between the two sets. Mean GPU decode
rose from 1400.6 ms to 1503.3 ms (+7.33%), while prefill rose from 671.6 ms to
717.5 ms (+6.84%). The NPU workload remained correct at cosine 1.0 and maximum
absolute error `9.54e-6`, but its own warm dispatch rose from 8.318 ms alone to
9.183 ms under overlap (+10.40%).

This proves simultaneous GPU/NPU execution works; it also quantifies shared
memory/fabric contention. A Gen5 expert-token costs about 0.459 ms under that
load. Putting even one selected expert per token on XDNA2 at every target layer
would add a dependency boundary at all 43 layers and cannot outrun the roughly
1.75 ms mean total verifier budget per layer. Do not implement that partition.
The already-separated speculative head is not a useful fallback: its entire
measured budget is only 1.7 ms and its fused GPU graph is dominated by the
4096x129280 target LM head, a shape the current fixed AIE kernels do not
support. The NPU remains useful as a zero-copy-capable experimental coprocessor,
but no measured serving-critical stage currently has a positive offload
budget.

The layer-major prefill shared expert was checked as the final plausible
serving role because every token uses the same weights. A cached dense Gen5
run with five tokens and one expert passed at cosine 1.0 and maximum absolute
error `2.38e-6`, but took 2.171 ms, or 0.434 ms per token. That is only about a
15% per-row improvement over the 0.513 ms single-row dense result. Repeating it
over 43 layers would spend roughly 18.7 ms per prompt token on the shared
expert alone; the reference all-GPU sparse-prefill result completes the entire
model in about 4 ms per token. Larger sequential XRT runlists cannot close that
gap. A useful prefill role would require a genuinely batched AIE matrix kernel
that reuses resident weights across token columns, not another extension of
Gen5's one-row expert commands.

### Gen6 batched shared expert

Gen6 implements that missing weight-reuse property for a fixed four-token
batch. Its 4x8 spatial graph consumes four BF16 activation rows together. Each
core decodes a 128x64 ROCMFP2 tile once, reuses the decoded BF16 tile for all
four rows, retains four independent FP32 accumulators, and carries the four
hidden rows through the same fused clamped-SwiGLU and two-pass down projection.
The host submits one XRT command for the complete four-row shared expert.
`EMBER_XDNA_KERNEL_GEN=6` accepts exactly four tokens and one expert so an
unsupported runtime shape fails closed instead of silently losing the reuse.

The current engine ABI supplies only cold routed experts to this provider; the
model's real shared expert remains part of the hot GPU graph. Consequently the
four-row/single-expert shape is currently reachable through the packaged
validator, not normal DeepSeek top-k serving. Selecting Gen6 for the existing
top-k-4 serving path will fail closed. An engine seam for shared-expert offload
was deliberately not added after the hardware timing showed that even the
improved kernel cannot meet the GPU prefill budget.

The topology follows the official MLIR-AIE whole-array examples' useful
pattern: split output ownership across cores, broadcast a weight stream over
the rows which reuse it, and retain partial results at the owning core. The
low-level replay keeps the AIE-RT completion-token rule discovered during
Gen5: every MM2S task which is subsequently waited upon requests a completion
token.

On the physical Strix Halo, three packaged dense four-token/one-expert validator
runs measured 247.218-248.089 ms for the CPU reference, 26.603-26.697 ms cold
including packing, and 0.840-0.853 ms averaged over 100 cached hardware runs.
All passed at cosine 1.0 with maximum absolute error `2.38e-6`. The warm result
is 0.210-0.213 ms per token, about 2.1x faster per row than Gen5's five-row
shared-expert runlist. This is the first AIE kernel in the prototype to obtain
a material gain from batching the arithmetic rather than only batching host
submissions.

That kernel gain is still not an inference win. Repeating 0.211 ms over all 43
target layers would spend about 9.1 ms per prompt token on the shared expert
alone, while the all-GPU reference completes the entire sparse model in about
4 ms per prompt token. Gen6 therefore remains a packaged experiment and Gen4
remains the provider default.

A follow-up AIE2P `mmul<4,8,8>` BF16 implementation compiled and routed into a
valid xclbin, but its command timed out on the physical NPU. That variant was
rejected and is not shipped; Gen6 retains the hardware-proven vector
accumulation above. Revisit native matrix instructions only with an isolated
tile-level layout/conformance test, not in the full expert graph.

The AIE-RT `release/main_aig` task-queue and repeat-count APIs motivated one
last Gen6 controller experiment. Collapsing the two 16-repeat hidden replays
into one 32-repeat task read beyond the intended staging cycle and produced
non-finite output. Queueing two 16-repeat tasks without the intermediate wait,
issuing a completion token only from the second, stayed finite but failed the
reference at cosine 0.9721 and maximum absolute error 1.33. Its 0.835 ms warm
time was indistinguishable from the exact 0.836 ms baseline. Keep the separate
token-producing task and `dma_wait` for each down-projection group: descriptor
reuse and the shallow broadcast FIFO make that wait a correctness boundary,
not useful host overhead.

### Fused-verifier shared-expert budget

The q-wide DSpark verifier was then measured by diagnostic graph surgery, not
kernel-name inference. `DFLASH_DS4_PROFILE_VERIFY_FFN=routed-only` removes the
shared branch and `shared-only` removes the routed branch from the opt-in fused
verifier. Both modes deliberately change model output and are non-authoritative;
normal inference defaults to `full`. Telemetry records fused compute separately
for q=2, q=3, and q=4 so changed acceptance cannot masquerade as a branch-cost
change.

On the physical gfx1151, with adaptive width and the profitability scheduler
disabled for the profiling request, per-call whole-model GPU compute was:

| verifier width | full | routed-only | shared-only | shared marginal |
| --- | ---: | ---: | ---: | ---: |
| q=2 | 62.9 ms | 54.6 ms | 40.9 ms | 8.3 ms |
| q=3 | 74.0 ms | 65.8 ms | 45.2 ms | 8.2 ms |
| q=4 | 81.7 ms | 76.2 ms | 48.9 ms | 5.5 ms |

The marginal column is `full - routed-only`: all 43 GPU shared experts cost
only about 0.13 ms per layer at q=4 inside the fused graph. The separately
measured Gen6 NPU path costs about 0.84 ms per layer, or 36 ms across the model,
before adding a production-weight decoder. More importantly, the production
shared tensors are `GGML_TYPE_Q4_0_ROCMFP4_FAST` (type 101), while Gen6 accepts
`GGML_TYPE_Q2_0_ROCMFP2` (type 107). A real shared-expert port therefore needs
a new FP4 packer and AIE kernel; it cannot reuse Gen6 weights directly.

Do not build that FP4 serving port under the current architecture. Even a
perfectly overlapped NPU branch can recover at most the 5.5 ms q=4 marginal,
roughly a 1-2% request-level ceiling in the measured 96-token run. Producing
each next layer's hidden state also requires 43 dependency barriers, which
would split the whole-model GPU graph that made the marginal so small. That is
before the already measured 7.33% GPU slowdown under concurrent NPU memory
traffic. Reconsider only if a future engine exposes a coarse independent
branch whose measured removed GPU time is materially larger, or if hardware
supports device-side cross-engine scheduling without per-layer host barriers.

### Coarse DSpark proposal seam

The next prototype moves the synchronization boundary from a target layer to a
complete DSpark support-model forward. This follows the useful parts of
[the bare-metal Ryzen NPU GPT-2 work](https://arxiv.org/html/2504.03083v1): keep
the AIE program resident, specialize the fixed shapes, double-buffer DMA and
compute, and change only runtime parameters instead of reconfiguring the array.
That paper measured full-array reconfiguration as 3.5x slower on the first
invocation of a new shape and identified the CPU-to-NPU-to-CPU round trip at
each kernel as a major remaining cost. Its conclusion also matches Ember's
Gen5/Gen6 measurements: a whole data-flow pipeline is the path to NPU utility;
many individually offloaded kernels are not.

`dspark_draft_compute_xdna.h` now defines an asynchronous provider ABI for one
complete three-layer draft forward. `submit()` must retain its input before it
returns, and the returned job can be waited or cancelled later. The output is
the normalized five-row hidden block plus the separate pre-output-norm state
used by the confidence head. Ember intentionally keeps the tied target LM,
Markov, and confidence heads outside the provider: this avoids duplicating the
129280-wide target output matrix in the first NPU artifact and leaves only a
short GPU head between NPU proposal completion and target verification.

The monolithic DSpark loop is wired through this provider first. It currently
submits and collects immediately, which is a correctness/qualification stage,
not yet an overlap claim. Optional failure falls back to the existing GPU
drafter; `DFLASH_DSPARK_XDNA_REQUIRED=1` fails closed. Configuration is:

```text
DFLASH_DSPARK_XDNA_PLUGIN=/usr/local/lib/libember_xdna_dspark.so
DFLASH_DSPARK_XDNA_REQUIRED=1
DFLASH_DSPARK_XDNA_GPU_MAIN=1
```

`DFLASH_DSPARK_XDNA_GPU_MAIN=1` selects the Gen13 placement boundary. Ember
runs the drafter's Q8_0 `main_proj` and `main_norm` as a small cached GPU graph,
then submits only the resulting `[ctx_len, n_embd]` `main_context` to the NPU
provider instead of the raw `[ctx_len, 3*n_embd]` captured features. The v1
request ABI adds this as a size-guarded tail field, so providers can continue
to consume the original prefix. The first implementation uses host-visible
staging between HIP and XRT; a shared BO/import path can replace that copy
without changing the provider's mathematical boundary.

GPU-free coverage loads a mock provider, submits two session proposals, mutates
the caller's raw and preprojected buffers after submission, and waits in reverse
order. This proves the lifetime, alternate-input, and session-isolation contract
without XRT. The shipped XRT module does not yet export the DSpark symbol: a
hardware provider needs the complete fixed-shape draft AIE graph and
trained-weight packer before the environment variables are usable in
production.

### Gen7 ROCMFP4 draft-expert primitive

Gen7 supplies the first trained-format building block for that graph. The
DSpark draft has three MoE layers, 256 experts, six routed experts per row, one
shared expert, a five-row block, and 4096/2048 expert dimensions. Its gate,
up, and down tensors are `GGML_TYPE_Q4_0_ROCMFP4_FAST` (type 101), not the
target prototype's ROCMFP2 type. Each 32-weight block is 16 signed-codebook
nibble bytes plus one UE4M3 scale byte.

The host packer losslessly permutes those blocks into 128x64 AIE tiles. It
keeps the nibble plane compact and expands only the scale to exact BF16, so a
packed block costs 18 bytes instead of 17 rather than doubling the weights to
int8. On AIE2P the microkernel vector-unpacks the nibbles, maps codebook
magnitudes `{0,1,2,3,4,6,8,10}`, applies the sign bit and BF16 scale once, and
reuses the decoded tile across all five draft rows. The resident 4x8 graph then
performs gate/up, exact clamped SwiGLU, and both down-projection groups without
an intermediate host round trip.

`expert_v7_4096x2048x4096_b5.{xclbin,insts}` and
`ember-xdna-rocmfp4-validate` are packaged in the `release-xdna` image. The
validator is deliberately separate from the ROCMFP2 target provider while the
whole draft graph is incomplete. It covers all 15 nonzero signed codebook
values, the complete projection shapes, five independent rows, and the fused
epilogue.

On the physical Strix Halo, the validator passed with maximum absolute error
zero and cosine 1.0. Twenty cached runs averaged 0.898 ms for the complete
five-row expert, with 14,155,776 packed weight bytes. The server stayed active
and healthy during the test. This is close to Gen6's roughly 0.84 ms four-row
FP2 result, proving that compact signed FP4 decode is not the limiting cost.

The result does **not** justify launching the dense Gen7 graph for every routed
expert. Five tokens select up to 30 routed expert rows per layer, and most
expert ids can be unique; a one-expert/five-row launch would compute four
unselected down-projection rows in that common case. Gen8 addresses this
selection waste while keeping the same coarse whole-draft boundary.

### Gen8 route-masked resident-expert primitive

Gen8 carries each row's router weight in the existing activation parameter
packet. Gate/up still consume the five rows because the fixed DMA stream sends
that packet last, but the fused SwiGLU writes an exact-zero BF16 hidden row for
tokens that do not select the current expert. The down microkernel detects that
zero tile and skips its MACs. For selected rows it applies the router weight
before the linear down projection, so the output is already the weighted
partial that the draft layer must accumulate.

Two more obvious control designs failed and are deliberately not shipped. A
separate routing FIFO exceeded the two shim-to-array DMA channels already used
by activations and weights. Holding the activation parameter object across the
hidden replay compiled but deadlocked on hardware because it prevented the
sequenced object FIFO from advancing. Reusing the parameter packet normally
and encoding inactive rows in the hidden data keeps the completion-token and
descriptor-reuse boundaries established by Gen5/Gen6.

The XRT host proof uses the public sub-buffer constructor to retain many packed
experts in one parent BO and select an expert by changing only the run's weight
view. A single runlist can therefore dispatch all unique experts without
repacking or synchronizing weights. `EMBER_XDNA_ROUTE_ROWS` selects the active
rows, `EMBER_XDNA_ROUTE_EXPERTS` repeats the route-aware command, and
`EMBER_XDNA_DISTINCT_WEIGHTS=1` makes every command address a different
resident expert-sized slab.

Physical Strix Halo results were exact (`max_abs=0`, cosine 1.0):

| Gen8 workload | warm sequence | per expert |
| --- | ---: | ---: |
| one expert, five active rows | 0.984 ms | 0.984 ms |
| one expert, one active row | 0.464 ms | 0.464 ms |
| 30 one-row experts, cache-hot address | 11.788 ms | 0.393 ms |
| 30 one-row experts, 30 distinct resident slabs | 11.786 ms | 0.393 ms |

The distinct test retained 424,673,280 packed weight bytes. Each slab held the
same structured correctness fixture so the final output stayed comparable, but
every command used a different page-aligned address. Its parity with the
same-address run rules out a cache-hot replay artifact and proves the dynamic
expert-address pattern needed by a resident 256-expert layer. The worst-case
routed portion is therefore about 11.8 ms per layer. The later trained Q8
shared-expert measurement adds 7.14 ms across the three-layer draft, putting
draft MoE near 43.5 ms including the three CPU reads/accumulations. That is
slower than the roughly 5 ms GPU drafter in isolation, but it fits beneath the
measured 63--82 ms GPU verifier window with roughly 19--38 ms left for
attention, normalization, routing, HC, and residual work. It remains a
cross-session throughput hypothesis, not an end-to-end acceleration claim.

`expert_v8_4096x2048x4096_b5.{xclbin,insts}` is packaged with Gen7 in the
opt-in `release-xdna` image.

### Gen9 route plan and weighted accumulation

`rocmfp4_route_plan.cpp` implements the deterministic host-control step between
the score router and Gen8. It accepts token-major top-k IDs and weights, rejects
invalid IDs, duplicates, negative/non-finite weights and empty plans, groups
collisions across tokens, and emits one expert-ID-ordered five-row mask per
unique resident expert. GPU-free tests cover collisions and the 30-unique
worst case. The inputs match DS4's router contract: the graph computes
`sqrt(softplus(logit))`, adds the optional bias only for top-k selection, then
normalizes the unbiased selected values and applies the model's 1.5 scale.

The hardware validator gives each planned run its own page-aligned input and
output sub-buffer. It executes the 30 commands in one XRT runlist, synchronizes
the 2.4 MiB output parent once, and accumulates every weighted expert partial
into the five final rows on the CPU. Two physical Strix Halo runs, including
the exact validator binary extracted from the release image, were exact
(`max_abs=0`, cosine 1.0): 11.814--11.824 ms NPU sequence time plus
0.247--0.305 ms for the single read/sum. The test uses six weights per token
whose sum is 1.5 and a different resident weight address per expert, so it
exercises the complete route-plan semantics rather than merely replaying
independent commands.

CPU accumulation is retained for the first trained differential because its
roughly 0.9 ms cost across three draft layers is small relative to the NPU MoE
work. An AIE accumulation stage is worthwhile only after trained outputs pass
and the complete provider shows that this read is on the critical path. The
trained-data gate now passes. `rocmfp4_model_weights.cpp` bounds-checks the
DSpark GGUF architecture, layer/expert metadata, type-101 tensor shapes, full
tensor extents, and requested expert IDs before using `pread` to extract only
the three selected 4,456,448-byte projection slices. It retains those raw
slices for the independent scalar reference and losslessly packs each expert
into its 14,155,776-byte resident image. A sparse 3.2 GiB GGUF fixture gives
this path GPU-free coverage without materializing the unused tensor pages.

The packaged validator read the deployed 10 GiB DSpark draft directly. Three
five-row runs covered layer 0/expert 0, layer 1/expert 127, and layer 2/expert
255. All had cosine 1.0; maximum absolute errors were `1.14e-5`, `1.34e-5`,
and `1.53e-5`, with warm times of 0.972, 1.000, and 0.909 ms respectively.
The layer-0 worst-case route plan then loaded and addressed 30 different
trained experts, applied the six normalized weights for each of five tokens,
and matched the accumulated scalar reference at `9.54e-6` maximum error and
cosine 1.0. Its NPU sequence was 11.885 ms, followed by 0.234 ms for the one
2.4 MiB read and CPU accumulation. Production remained active and healthy,
and the host IOMMU group reported translated `DMA-FQ` mode.

### Gen10 compensated Q8 draft shared expert

The deployed DSpark manifest establishes a separate quantization path for the
dense/shared branch. Its three layers store
`blk.N.ffn_{gate,up,down}_shexp.weight` as standard GGML Q8_0 (type 8), while
the routed experts above are ROCMFP4_FAST (type 101). The strict loader checks
the `deepseek4-dflash-draft` architecture, three-layer metadata, tensor names,
type, dimensions, extents, and file bounds before packing any bytes. A sparse
fixture covers those checks without constructing a 10 GiB test file.

Two numerically attractive shortcuts were hardware-rejected. The first
expanded every Q8 weight once to one BF16 value (50,331,648 bytes per shared
expert). Its synthetic fixture was exact at 1.478 ms, but trained layer 0 had
`0.108` maximum error and cosine `0.9999946`; the three trained layers ranged
from `0.0699` to `0.1169` maximum error. Rounding the FP16 block scale into one
BF16 product is therefore not release quality. The second retained int8 codes
and exact F32 scales (28,311,552 bytes). It appeared compact and timed at
1.114 ms, but failed even the synthetic fixture at `0.140` maximum error and
negative cosine. Peano disassembly showed why: AIE2P vector FP32 multiplication
was lowered through BF16 operand conversions, rounding every 32-input partial
before scale application. Exact storage did not imply exact executed math.

The accepted kernel stores each dequantized Q8 weight as a BF16 high term plus
a BF16 residual. Both 128x64 planes feed native BF16 MACs into the same FP32
accumulator, avoiding the unsupported vector-FP32 operation. The representation
is 100,663,296 bytes per expert and requires a single-buffered weight FIFO to
fit each 64 KiB compute tile; activations and results remain double buffered.
The complete 4x8 graph retains gate/up, exact clamped SwiGLU, hidden replay,
both down groups, and all five draft rows under one XRT command.

Sequential physical Strix Halo results from the packaged artifact were:

| Q8 shared expert | warm sequence | maximum absolute error | cosine |
| --- | ---: | ---: | ---: |
| synthetic | 2.374 ms | `1.49e-8` | 1.0 |
| trained layer 0 | 2.383 ms | `4.94e-4` | `0.9999999999` |
| trained layer 1 | 2.380 ms | `5.40e-3` | `0.9999999921` |
| trained layer 2 | 2.381 ms | `4.41e-3` | `0.9999999972` |

All runs satisfy the unchanged cosine `>=0.99999` and maximum-error `<=0.01`
gate. The three trained shared experts total 7.144 ms. Adding Gen9's measured
35.655 ms routed sequences and about 0.70 ms of CPU reads/weighted sums puts
the measured draft-MoE subtotal at about 43.5 ms for five proposal rows. This
still leaves a positive 19--38 ms overlap budget under the measured q=2..4
GPU verifier, but only a complete provider and aggregate two-session benchmark
can determine whether attention/HC/dense work and shared-fabric contention fit.

`shared_q8_v2_4096x2048x4096_b5.{xclbin,insts}` and
`ember-xdna-q8-validate` are packaged in `release-xdna`; the rejected variants
are not selected by the image (the one-plane BF16 baseline remains an explicit
research build target). The graph topology follows the same full-array
principles as IRON's BF16 GEMM, MHA, RMSNorm, and transformer examples, but its
compensated Q8 numerical contract is Ember-specific. AIE-RT remains
compiler-side: `aiecc.py` emits the immutable instruction stream, while the
runtime image executes it through XRT rather than linking a second
`libxaiengine` transaction generator.

Run the trained Q8 check with:

```bash
docker run --rm --privileged --ipc=host --ulimit memlock=-1 \
  -v /path/to/models:/models:ro --entrypoint \
  /usr/local/bin/ember-xdna-q8-validate ember:xdna-local \
  /usr/local/share/ember/xdna2/shared_q8_v2_4096x2048x4096_b5.xclbin \
  /usr/local/share/ember/xdna2/shared_q8_v2_4096x2048x4096_b5.insts \
  /models/dspark-draft.gguf 0
```

### Gen11 task-blocked Q8 attention projections

The same compensated arithmetic now covers the five Q8_0 matrices in each
DSpark MLA layer. The generic graph activates only the required AIE rows for
512- and 1024-wide outputs, then distributes wider output groups over the full
4x8 array. Its weight stream is packed column-major so one shim task feeds one
column without the 8 MiB stride produced by a group-major layout.

Compiler validation exposed two AIE DMA constraints that are easy to miss at
the IRON level but explicit in the AIE-RT descriptor contract: a shim stride
cannot exceed one MiB, and the compiler must not coalesce a contiguous transfer
past the same ceiling. The final pack divides each column into four-K-tile
tasks and inserts two BF16 sentinels between tasks. The four-byte-aligned gap
prevents coalescing; it adds at most 1 KiB to a 128 MiB packed projection and
does not enter the compute stream. No AIE-RT library is linked at runtime.

Layer-0 trained results on the physical Strix Halo were:

| DSpark tensor | K x N | warm sequence | packed bytes | maximum error | cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attn_q_a` | 4096 x 1024 | 0.392 ms | 16,777,472 | `2.00e-5` | 1.0 |
| `attn_q_b` | 1024 x 32768 | 2.501 ms | 134,218,752 | `1.04e-5` | 1.0 |
| `attn_kv` | 4096 x 512 | 0.261 ms | 8,388,864 | `1.38e-5` | 1.0 |
| `attn_output_a` | 4096 x 8192 | 2.514 ms | 134,218,752 | `2.96e-5` | 1.0 |
| `attn_output_b` | 8192 x 4096 | 2.500 ms | 134,218,752 | `4.20e-5` | 1.0 |

The five trained projections total 8.168 ms per same-shaped layer, replacing
the earlier bandwidth-only estimate. Repeating that measured shape cost over
three layers adds about 24.50 ms to Gen10's 43.5 ms MoE subtotal, for roughly
68.0 ms before MLA dot/softmax/context, RMSNorm, the F32 HC transforms, router,
and tail collapse. This materially narrows the viable coarse-overlap target:
q=2's 62.9 ms verifier cannot hide the current draft floor; q=3 leaves about
6 ms and q=4 about 14 ms. Therefore Gen11 is an accuracy- and bandwidth-proven
building block, not yet evidence of an end-to-end throughput win. Fusing the
two small 4096-input projections and measuring the remaining attention/HC path
are the next gates.

The five `q8_projection_v3_*_b5.{xclbin,insts}` artifacts and
`ember-xdna-q8-projection-validate` are packaged in `release-xdna`. For example:

```bash
docker run --rm --privileged --ipc=host --ulimit memlock=-1 \
  -v /path/to/models:/models:ro --entrypoint \
  /usr/local/bin/ember-xdna-q8-projection-validate ember:xdna-local \
  /usr/local/share/ember/xdna2/q8_projection_v3_1024x32768_b5.xclbin \
  /usr/local/share/ember/xdna2/q8_projection_v3_1024x32768_b5.insts \
  1024 32768 /models/dspark-draft.gguf blk.0.attn_q_b.weight
```

### Gen12 weight-reusing 128-row Q8 context GEMM

Gen11 measured the five proposal rows but did not include DSpark's dense
context path. The draft first projects up to 128 concatenated target-layer
features through `dflash.fc.weight`, and every draft layer projects the same
context rows through `blk.N.attn_kv.weight`. The deployed GGUF stores both as
ordinary Q8_0. Treating those operations as five-row GEMVs materially
under-counted the draft cost.

Gen12 follows the full-array BF16 GEMM topology in AMD IRON's
`aie_kernels/aie2p/mm.cc`: four AIE rows each own 32 input rows, eight columns
each own 64 output columns, and the `mmul<4,8,8>` kernel expands two M blocks
by two N blocks to keep four FP32 accumulators live. Activations multicast
across each compute row and corrected Q8 weights multicast down each column.
Each dequantized weight remains a BF16 high term plus BF16 residual, so this
changes reuse and scheduling without weakening Gen10/11's trained numerical
contract.

The larger graph exposed three additional descriptor rules in the pinned
MLIR-AIE/AIE-RT flow:

- aggregating four compute-row outputs in the memory tile is necessary to stay
  within each shim's input-channel count;
- no DMA dimension may encode 1024 elements, so 8--32 KiB objects are expressed
  as 512-element subdimensions;
- the shim repeat field accepts at most 64 objects. K is consequently divided
  into 32-tile runs, and a four-byte sentinel after every corrected weight tile
  prevents adjacent 32 KiB objects from coalescing into an illegal transfer.

For the eight-output-group main projection, the 3 MiB activation is replicated
in host-visible staging for each output group (about 25 MiB total). This makes
the descriptor sequence affine and is far cheaper than duplicating or
re-reading the 201 MiB corrected weights.

Physical Strix Halo results against trained tensors were:

| DSpark tensor | M x K x N | warm sequence | packed bytes | maximum error | cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `blk.0.attn_kv.weight` | 128 x 4096 x 512 | 0.673 ms | 8,389,632 | `2.19e-5` | 1.0 |
| `dflash.fc.weight` | 128 x 12288 x 4096 | 13.929 ms | 201,351,168 | `5.53e-5` | 1.0 |

The context-KV graph processes 25.6 times as many rows as Gen11's five-row
0.261 ms projection in only 2.58 times the sequence time, roughly a 9.9x
per-row throughput improvement from weight reuse. The main projection plus
three layer-local context-KV projections nevertheless costs 15.95 ms. Adding
that previously missing work to Gen11's roughly 68.0 ms subtotal produces an
83.95 ms full-context draft floor before HC, RMSNorm, router, attention
reductions, and tail collapse. It cannot be completely hidden by the measured
q=4 verifier window of about 81.7 ms, so Gen12 is a successful kernel but an
end-to-end scheduling rejection in its current compensated form. Reducing the
13.9 ms main projection is the next performance gate.

Both `q8_gemm_m32_v4_{4096x512,12288x4096}_b128.{xclbin,insts}` artifacts and
`ember-xdna-q8-gemm-m32-validate` are packaged in `release-xdna`. A trained
main-projection check is:

```bash
docker run --rm --privileged --ipc=host --ulimit memlock=-1 \
  -v /path/to/models:/models:ro --entrypoint \
  /usr/local/bin/ember-xdna-q8-gemm-m32-validate ember:xdna-local \
  /usr/local/share/ember/xdna2/q8_gemm_m32_v4_12288x4096_b128.xclbin \
  /usr/local/share/ember/xdna2/q8_gemm_m32_v4_12288x4096_b128.insts \
  12288 4096 /models/dspark-draft.gguf dflash.fc.weight
```

### Gen13 AIE-RT/IRON BFP16 MMUL rejection

AMD IRON's current AIE2P GEMM enables
`AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` by default. The corresponding
[AIE-RT `release/main_aig` branch](https://github.com/Xilinx/aie-rt/tree/release/main_aig)
exposes the descriptor/runtime layer used by that graph. On Strix Halo the
emulation changes the BF16 matrix tile from native `mmul<4,8,8>` to
`mmul<8,8,8>`. Gen13 reproduces that choice in `q8_gemm_m32_v5` while keeping
Gen12's compensated BF16 high-plus-residual weight representation and FP32
accumulators.

The trained result is fast but does not pass Ember's unchanged numerical gate:

| DSpark tensor | native Gen12 | BFP16 Gen13 | maximum error | mean error | cosine | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `blk.0.attn_kv.weight` | 0.673 ms | 0.289 ms | `0.07212` | `0.01158` | 0.9999818 | fail |
| `dflash.fc.weight` | 13.929 ms | 4.621 ms | `0.08729` | `0.01588` | 0.9999826 | fail |

The maximum-error limit is 0.05 and the cosine limit is 0.99999. BFP16 MMUL
therefore is not packaged, selected, or used by the provider. The v5 source,
build target, and validator mode remain as reproducible negative evidence; run
the validator with `EMBER_XDNA_Q8_BFP16_MMUL=1` against a separately built v5
artifact.

This rejection also changes the heterogeneous placement decision. Moving the
201 MiB `dflash.fc.weight` projection to the NPU is unnecessary: the GPU can
compute that one main-context projection as a short dependency pre-stage, then
the NPU can consume its normalized `main_x` while the GPU verifies another
resident session. Keeping the NPU on native compensated Q8 removes 13.929 ms
from its measured draft subtotal, reducing the known floor from 83.95 ms to
about 70.02 ms before the GPU pre-stage, HC transforms, norms, routing,
attention reductions, and tail collapse.

The packaged `ember-dspark-main-bench` measures the exact graph topology and
host-visible handoff used by `deepseek4_dspark_project_main_context()` without
allocating the 85 GiB target model. Physical gfx1151 results with the trained
Q8_0 projection were:

| context rows | upload mean | GPU graph mean | download mean | staged total mean | staged total median |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.014 ms | 0.278 ms | 0.011 ms | 0.303 ms | 0.300 ms |
| 128 | 0.109 ms | 0.697 ms | 0.034 ms | 0.841 ms | 0.833 ms |

This is 16.6x faster than Gen12's numerically valid 13.929 ms NPU main
projection, including both host transfers, and 5.5x faster than the rejected
4.621 ms BFP16 result. The measured draft floor is therefore about 70.86 ms,
leaving about 10.84 ms under the measured 81.7 ms q=4 verifier window for the
missing work. Upload plus download is only 0.144 ms at 128 rows. A direct
GPU/XRT buffer handoff can recover at most that amount here, and projecting
only four appended rows can recover at most roughly 0.54 ms. Neither is the
next performance gate; completing and measuring the asynchronous NPU body is.

Run the isolated placement check from the opt-in image with:

```bash
docker run --rm --privileged --ipc=host --ulimit memlock=-1 \
  -v /path/to/models:/models:ro --entrypoint \
  /usr/local/bin/ember-dspark-main-bench ember:xdna-local \
  /models/dspark-draft.gguf 20
```

### Gen14 CPU control/reduction body

The heterogeneous boundary leaves only operations that are too small or too
reduction-heavy for the current AIE graphs on the CPU. Gen14 implements that
side as a provider-independent library: weighted RMSNorm, the exact four-stream
HC sigmoid/Sinkhorn pre/post/tail equations, unbiased sqrt-softplus router
top-k, and full-visibility MLA score/sink-softmax/context reduction with tail
RoPE. A persistent worker pool preserves each dot product's accumulation order
while parallelizing independent HC rows, experts, and query heads. The same
code is covered by the GPU-free `xdna_dspark_cpu_ops` test.

The packaged shape benchmark executes three layers of both HC boundaries,
all 15 layer-local norms plus the final norm, three router selections, three
5-query by 133-key attention
reductions, and the final HC collapse/norm. It excludes the large projections
and expert graphs already measured on XDNA2. On the production Ryzen CPU, the
warm results were:

| CPU workers | three-layer control/reduction body |
| ---: | ---: |
| 4 | 18.057 ms |
| 8 | 10.073 ms |
| 12 | 8.885 ms |
| 16 | 8.087 ms |
| 20 | 7.853 ms |
| 24 | 8.043 ms |
| 32 | 8.028 ms |

Sixteen workers remain the default because the GPU verifier runs concurrently
and needs package/fabric headroom; `EMBER_DSPARK_CPU_THREADS` permits a measured
1--64 override. At that default, the projected draft body is about 78.95 ms
(`70.86 + 8.09`), leaving only about 2.75 ms under the 81.7 ms q=4 verifier
window for provider orchestration and shared-fabric interference. This is the
first complete shape-based result that fits, but the margin is too small to
claim an end-to-end win without the aggregate concurrent run.

Run the isolated CPU placement check with:

```bash
EMBER_DSPARK_CPU_THREADS=16 ember-dspark-cpu-bench 10
```

Run the same trained check from the opt-in image with:

```bash
EMBER_XDNA_ROUTE_PLAN=1 docker run --rm --privileged --ipc=host \
  --ulimit memlock=-1 -e EMBER_XDNA_ROUTE_PLAN \
  -v /path/to/models:/models:ro --entrypoint \
  /usr/local/bin/ember-xdna-rocmfp4-validate ember:xdna-local \
  /usr/local/share/ember/xdna2/expert_v8_4096x2048x4096_b5.xclbin \
  /usr/local/share/ember/xdna2/expert_v8_4096x2048x4096_b5.insts \
  /models/dspark-draft.gguf 0
```

Without `EMBER_XDNA_ROUTE_PLAN=1`, `EMBER_XDNA_MODEL_EXPERT` selects one
expert (default 0) and the final positional argument selects layer 0--2. The
remaining performance gate is the complete asynchronous three-layer draft
provider; real router selections will then be captured from that provider's
input rather than synthesized by the standalone route planner.

The scheduling target is deliberately cross-session and coarse: while the GPU
verifies session A (roughly 63-82 ms for q=2..4 in the measurements above), the
NPU prepares session B's five-row draft hidden block. The CPU owns admission,
proposal-job bookkeeping, and completion. This can hide an NPU drafter much
slower than the current roughly 5 ms GPU drafter without splitting the fused
target graph. It still must beat the measured shared-fabric contention in an
aggregate two-session benchmark; UMA removes a copy opportunity, not bandwidth
competition or XRT/ROCm allocation-coherence requirements.

Before treating the path as usable on gfx1151 hardware:

1. Keep the passing trained ROCMFP4 expert and weighted-runlist gate in CI and
   hardware qualification; every output lane is compared and failures close.
2. Compare the complete three-layer draft provider (attention, HC, norms,
   routing, shared and routed experts) to the HIP drafter at its hidden-state
   and head boundaries before enabling asynchronous scheduling.
3. Run Ember's differential validator with
   `DFLASH_MOE_XDNA_REQUIRED=1`, snapshot restore, disk round-trip, two resident
   sessions, and DSpark when configured.
4. Measure end-to-end tok/s and per-token latency against HIP-only, with cold
   packing separated from warm steady state. Record provider launch/sync time,
   total package power, GPU slowdown while XDNA runs, thermals, cache hit rate,
   and `xrt-smi` partition activity.
5. Keep Gen5 target-MoE offload opt-in and Gen6 validator-only. Gen5's trained
   baseline passed the initial token check, but its warm end-to-end decode
   reached only 10.20 tok/s versus 22.65 tok/s for production; Gen6 improves
   the standalone shared expert but still exceeds the entire GPU prefill
   budget before accounting for any other layer work.
6. Preserve the rejected DSpark-drafter and per-layer-verifier results. A future
   serving partition must first identify a stage whose removed GPU time exceeds
   both its NPU execution and the measured shared-fabric penalty; zero-copy
   transport alone is not sufficient.

DeepSeek's HC update remains sequential at q=1 in this hybrid path. Raising a
prefill chunk limit without preserving every token boundary is not correct.
