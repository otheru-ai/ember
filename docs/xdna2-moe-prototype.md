# XDNA2 MoE expert prototype

This branch contains an experimental seam for running selected DeepSeek-V4
routed experts on the Strix Halo XDNA2 NPU while the existing HIP path retains
attention, routing logits, shared experts, and residual combination. It is off
by default and does not yet include a production XRT/IRON kernel.

The design follows AMD's published GPT-OSS-20B QMoE implementation: CPU code
performs top-K routing and groups tokens, then an accelerator evaluates only
the selected expert projections. See AMD's
[Strix/Halo QMoE case study](https://www.amd.com/en/developer/resources/technical-articles/2026/accelerating-gpt-oss-20b-on-amd-ryzen-ai-npus.html).

## What the prototype proves

- `MoeExpertCompute` can decline inefficient shapes, allowing the measured GPU
  path to retain any batch shape the provider cannot accelerate.
- DeepSeek-V4's hybrid graph now passes cold-expert work through the existing
  selected-expert abstraction instead of always passing a null backend.
- A provider is loaded at runtime through a versioned C ABI. Ember therefore
  has no compile-time XRT dependency and the ROCm release remains usable when
  the XDNA driver or provider is absent.
- Placement-local expert IDs are translated to model-global IDs before the
  provider sees them. A provider can load or requantize precisely the experts
  it owns directly from the GGUF.
- Optional provider failure disables the provider and replays the operation on
  the existing mmap-to-GPU path. `DFLASH_MOE_XDNA_REQUIRED=1` makes any load or
  runtime failure fatal for benchmarking and differential validation.

The GPU-free `xdna_moe_provider` test loads a mock shared object and verifies
ABI negotiation, shape gating, expert-ID translation, router weights, output,
and load errors. The mock deliberately performs ordinary host arithmetic; it
is not a performance implementation.

## Provider ABI

Implement `ember_xdna_moe_get_provider_v1` from
`engine/dflash/common/moe_expert_compute_xdna.h`. The returned vtable owns a
persistent provider context and receives `ember_xdna_moe_batch_v1` requests:

- F32 input activations shaped `[n_tokens, n_embd]`;
- global expert IDs and F32 routing weights shaped
  `[n_tokens, n_selected]`;
- an F32 output buffer for the already-weighted sum of routed expert outputs.

The provider receives the model path and architecture dimensions at creation.
This is intentional: the production provider must parse the GGUF, retain its
own NPU-resident expert cache, and either decode ROCMFP weights directly or
maintain a quality-gated NPU representation. Passing host tensor pointers
would incorrectly imply that ROCm and XRT share a coherent allocation API.

`test/mock_xdna_moe_provider.cpp` is the smallest complete provider example.

## Enable the prototype

Build Ember normally in the ROCm development container, then launch with:

```bash
DFLASH_MOE_XDNA_PLUGIN=/opt/ember/libember_xdna_moe.so \
DFLASH_EXPERT_BUDGET_MB=32768 \
./build-rocm/ember-dflash -m /models/model.gguf
```

`DFLASH_EXPERT_BUDGET_MB` must leave cold experts in the hybrid placement. If
all experts are assigned to the GPU, there is nothing for the provider to do.

Prototype controls:

| variable | default | purpose |
|---|---:|---|
| `DFLASH_MOE_XDNA_PLUGIN` | unset | Provider shared-object path; unset disables the feature |
| `DFLASH_MOE_XDNA_REQUIRED` | `0` | Fail instead of falling back if provider loading or execution fails |
| `DFLASH_MOE_XDNA_MIN_TOKENS` | `1` | Smallest batch sent to the provider |
| `DFLASH_MOE_XDNA_TRACE` | `0` | Log provider call, token, and wall-time totals at shutdown |

## Production-provider work

The stock Ryzen AI operator set supports W4A-BF16/W4A-FP16 matmuls, not the
model's affine fp2 ROCMFP representation. A production provider therefore
needs one of:

1. an IRON/AIE-MLIR kernel that decodes ROCMFP and performs the gate, up,
   SwiGLU, and down projections on XDNA2; or
2. a bounded INT4 cache of popular cold experts with a quant-quality report
   proving that its output remains inside the release gates.

The current DeepSeek hybrid path deliberately retains q=1 prefill because its
HC update is sequential. This prototype therefore exercises the provider at
q=1 by default. A later prefill-batching change must preserve HC state at every
token boundary and pass the differential validator independently; merely
raising a chunk limit is not correct.

The first hardware milestone should benchmark exact DS4 shapes at batch sizes
1, 2, 4, 8, 16, 32, 128, and 512. Record provider dispatch and synchronization
time, total memory bandwidth, GPU slowdown while XDNA runs, power, and thermal
steady state. Do not enable decode by default unless batch-1 end-to-end latency
wins after the GPU/NPU synchronization cost.

Before considering the provider deployable, run the existing differential
validator with `DFLASH_MOE_XDNA_REQUIRED=1`, including snapshot restore, disk
round-trip, two resident sessions, and DSpark when compatible. HIP-only output
and performance remain the release baseline.
