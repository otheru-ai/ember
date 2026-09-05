# Vendored engine

The engine is a vendored fork of [Luce-Org/lucebox](https://github.com/Luce-Org/lucebox),
pinned to upstream commit `8fd9584` from its `main` branch. It is distributed
as source so Ember can build against its stable C ABI without requiring a
network fetch.

| path | upstream basis | license | notice |
|---|---|---|---|
| `engine/ggml` | llama.cpp/ggml fork with ROCMFP kernels | MIT | `engine/ggml/LICENSE` |
| `engine/hip_compat` | ggml HIP compatibility layer | MIT (ggml) | `engine/ggml/LICENSE` |
| `engine/dflash` | DeepSeek4 backend and common subset | Apache-2.0 | `engine/dflash/LICENSE` |
| `engine/third_party/nlohmann` | nlohmann/json | MIT | `engine/third_party/nlohmann/LICENSE.MIT` |

## Local modifications

DRY-aware DSpark target sampling is a local Ember extension. The opt-in serial
fresh/restore path shares the ordinary request sampler, accepts only emitted
history, and advances grammar/structural hooks between selected rows. Strict
q=1 verification is its default; sampled q-wide verification is separately
opt-in and numerically approximate. See `docs/dspark-sampling.md`. The resident
NPU path retains its existing sampled-request fallback.

`engine/` is intentionally a fork, not an unmodified third-party mirror.
Local changes cover the C ABI bridge, gfx1151 ROCMFP integration, DSpark
scheduling, resident-session batching, the opt-in XDNA2 selected-expert and
asynchronous whole-draft provider seams, q-wide resident verification,
release-specific tests, and three audited ROCmFPX correctness ports (`00d54526`
allocation views, `5ed0d9ef` HIP norm support, and `a8b5fa90` ROCMFP2 CPU
OUT_PROD). HIP fast-math remains deliberately disabled per ROCmFPX `8e6277f8`'s
gfx1151 speculative-decode measurements. Before updating it, diff the candidate
upstream revision against `8fd9584`, preserve the license notices above, and
record the new upstream commit and any local divergence in this file.

The ROCMI4 storage/runtime and optional gfx1151 W4A4 path are manually ported
from the MIT-licensed `radicalgeek/ROCmFPX` lineage: exact format commit
`16d05b80f70b06b008da26bc1be7d36f116c61e4`, lossless int8 MMQ commit
`fef417b287240a573082bccf01705af67441e9be`, IU4 experiment commit
`659456f7a2e4bfc54657b5a91692c5ac05fa259c`, and the narrowed/off-by-default
gfx1151 gate `cef686ca09c1a9f276897b31b93bb621128b85fb` (documentation follow-up
`928ccb3eb6feedaa5c26c7e6a723852c14e44115`). The port is surgical because
Ember's engine is pruned and locally optimized; it intentionally keeps exact
int8 MMQ as the default. Canonical ROCMI4 owns GGUF file type 118. Ember's older
Q2 recipe metadata moved to 119/120 while its on-disk tensor type 107 remains
unchanged, preserving tensor-dispatched loading of already-published files.

ROCm-family MMQ dispatch evidence is also an Ember-owned diagnostic extension
in `ggml-cuda/mmq.cu`: the legacy
`DFLASH_ROCMI4_W4A8_DISPATCH_EVIDENCE` gate now labels the default ROCMI4 q8_1
DP4A kernel, the optional W4A8 variants, and the ROCmFP4-fast q8_1 DP4A kernel.
The model-free operator oracle requires that inner marker so a selector
fallback or wrong quant type cannot produce a vacuous green result; preserve it
across vendor refreshes.
`DFLASH_MMQ_SRC1_INVENTORY=1` is the companion off-by-default full-model
diagnostic: each dense MMQ dispatch records its activation dimensions, byte
strides, contiguity/view predicates, weight identity, and selected MMQ route.
This includes both ordinary dense dispatches and the pair-fused dense gate/up
path; a routed pair is labeled separately if encountered. It is
logging-only; when enabled it synchronously captures the stride-aware activation
range so production dynamic range can be compared with the synthetic oracle.
It exists to distinguish a live layout or value-domain difference from a
fixture that the shipped graph never constructs.

Ember pins the shape contract between `g` and `beta` with the host-testable
predicate in `ggml-cuda/gated_delta_net_layout.h`, asserted by
`gated_delta_net.cu`. Upstream indexes
both from one offset built from beta's outer strides while asserting only that
each is contiguous, which does not imply compatible shapes. The predicate
requires equal outer dimensions and, for a scalar gate, equal strides. KDA gates are intentionally
`[S_v, H, T, B]` rather than beta's `[1, H, T, B]`, so their strides differ and
the kernel's existing `gb_offset * S_v` supplies the required rescaling. The
caller's GDN graph uses the scalar form and reshapes both tensors to
`[1, n_heads, n_tokens, 1]`; the assertions pin that caller contract so a
future reshape fails loudly instead of silently misreading `g` at beta's
spacing. Preserve them across vendor refreshes.

## Pruned deployment scope

Ember preserves the upstream provenance above, but intentionally does not carry
the whole portable lucebox/ggml source matrix. This vendored snapshot is scoped
to the only supported deployment: Linux x86-64 on AMD Strix Halo, with one
gfx1151 HIP device and the on-package XDNA2 NPU. Non-AMD compute backends,
non-x86 CPU implementations, NVIDIA-only sampling, remote execution, layer
splitting, and multi-GPU peer/shard placement have been removed. The remaining
`ggml-cuda` directory name is an upstream compatibility detail: ROCm's HIP build
compiles those shared `.cu`/`.cuh` kernels directly, and they are load-bearing
for the gfx1151 path. Its RDNA4/gfx12 WMMA implementations, dispatch cases, and
tuning tables are removed: gfx1151 uses the distinct RDNA 3.5 fragment layout,
and retaining both made an unsupported architecture look testable. The public
split/peer-copy surface and RCCL integration are removed; a compile-time guard
also excludes the shared substrate's internal peer-copy branch.

The retained dflash source closure contains only the DeepSeek4 target. The
legacy non-DeepSeek target loaders, draft/DeltaNet object graph, platform
mmap shim, and their pre-tokenizers are removed. DeepSeek4 keeps a narrow CPU-embedding
helper and accepts only the checkpoint's required `tokenizer.ggml.pre` value,
`joyai-llm`. Legacy Laguna expert-remap/cache code, its custom CUDA combine op,
generic chat-family probing, and unused placement/load helpers are also absent.
The unused C++ prefix-cache policy, routing-statistics collector, and DDTree
prototype are removed as well; Ember owns prefix policy above the backend ABI,
and the shipped DeepSeek target implements chain verification only. The shared
SSM and gated-delta-net HIP kernels therefore retain only their live chain-mode
forms.
The allocation-free thinking-budget state machine is maintained as a
C-compatible header and exercised by a C test. Crash-breadcrumb storage and the
thread-safe loader diagnostic channel are also compiled as C. The C++ DeepSeek
backend consumes these APIs directly; these are the first low-risk orchestration
components migrated toward C without changing the tokenizer, graph builder, or
HIP kernels.
Runtime file and plugin paths are Linux/POSIX-only, matching the top-level
configure guard rather than carrying unreachable Windows fallbacks.

When importing a future engine fix, diff against upstream commit `8fd9584` and
port only files reachable by this retained HIP/x86-64 build. A broad vendor
refresh would reintroduce unsupported architectures and must be pruned and
revalidated before landing.
