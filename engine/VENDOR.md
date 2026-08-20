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

## Pruned deployment scope

Ember preserves the upstream provenance above, but intentionally does not carry
the whole portable lucebox/ggml source matrix. This vendored snapshot is scoped
to the only supported deployment: Linux x86-64 on AMD Strix Halo, with one
gfx1151 HIP device and the on-package XDNA2 NPU. Non-AMD compute backends,
non-x86 CPU implementations, NVIDIA-only sampling, remote execution, layer
splitting, and multi-GPU peer/shard placement have been removed. The remaining
`ggml-cuda` directory name is an upstream compatibility detail: ROCm's HIP build
compiles those shared `.cu`/`.cuh` kernels directly, and they are load-bearing
for the gfx1151 path. The public split/peer-copy surface and RCCL integration
are removed; a compile-time guard also excludes the shared substrate's internal
peer-copy branch.

When importing a future engine fix, diff against upstream commit `8fd9584` and
port only files reachable by this retained HIP/x86-64 build. A broad vendor
refresh would reintroduce unsupported architectures and must be pruned and
revalidated before landing.
