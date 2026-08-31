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

The Qwen3.8-Flash-Next text runtime also carries an Ember-owned C static-YaRN
policy/reference implementation. Its arithmetic follows Transformers revision
`36deb0b53ed0863f4b4dfdea23dcaec7f3df3701`; the exact factor-4, 262144-to-1M
recipe is pinned to the official model README revision
`f5d08274bafd880402bd16f5e3e6c514136ec06c`. This is an explicit operator
override because the checkpoint metadata remains ordinary RoPE. The local q=1
runtime uses the C reference for its QSA and indexer positions; future graph
paths should pass the same resolved parameters through ggml's existing
`ggml_rope_multi`/`GGML_ROPE_TYPE_IMROPE` implementation rather than adding a
new kernel. The 128-GiB memory planner remains authoritative and may reject the
official 1M recipe for otherwise valid released weights.

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

The Qwen3.8-Flash-Next format bakeoff also admits the existing
`Q3_0_ROCMFPX` type (GGML tensor type 104) for the mapped
`per_layer_token_embd.weight` row table. Its 32-weight/14-byte layout was
re-audited against `ciru-ai/ROCmFPX` revision
`112629f1ed1acc2e8071693fce83cc7f5070693a`; no broad vendor refresh was
performed. The tensor mix is provenance-pinned to
`agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-GGUF` revision
`9089b24dbed6e087a705201ba59a104575bda0b9`, but only as an experimental
recipe input. Ember's own intervention, quantization audit, quality, memory,
PLE-latency, and gfx1151 gates remain authoritative.

The off-by-default Qwen correctness diagnostic also extends
`GGML_CUDA_FORCE_CUBLAS` to suppress quantized MMVQ in the fusion predicate,
plain `mul_mat`, and `mul_mat_id`, so every quantized projection reaches the
dequantize-and-GEMM family. `DFLASH_CUBLAS_F32_REFERENCE=1` then forces the
existing cuBLAS fallback to dequantize quantized operands to F32 and emits
positive route evidence; it fails closed unless the force-cuBLAS build and
`GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` are both present. This explicit env is
necessary instead of `ggml_mul_mat_set_prec(GGML_PREC_F32)` because upstream's
`mul_mat_id` synchronous fallback zero-initializes each per-expert destination
and thereby silently drops the caller's precision request. Ordinary builds and
force-cuBLAS builds without the env retain their prior operand precision.

ROCMI4 MMQ dispatch evidence is also an Ember-owned diagnostic extension in
`ggml-cuda/mmq.cu`: the legacy `DFLASH_ROCMI4_W4A8_DISPATCH_EVIDENCE` gate now
labels the default q8_1 DP4A kernel as well as the optional W4A8 variants. The
model-free row-tail oracle requires that inner marker so a selector fallback
cannot produce a vacuous green result; preserve it across vendor refreshes.

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
legacy Qwen3.5 target loader, draft/DeltaNet object graph, platform mmap shim,
and Qwen pre-tokenizers are removed. DeepSeek4 keeps a narrow CPU-embedding
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
