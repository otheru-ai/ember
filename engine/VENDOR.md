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
scheduling, resident-session batching, and release-specific tests. Before
updating it, diff the candidate upstream revision against `8fd9584`, preserve
the license notices above, and record the new upstream commit and any local
divergence in this file.
