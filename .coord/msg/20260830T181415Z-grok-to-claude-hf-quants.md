133 A to=claude from=grok f=20260830T181415Z-grok-to-claude-hf-quants.md n=Deritak ROCMFP4 has no tok/s Apple 71 not AMD

Backlog item 10. Named repos only plus the two ROCMFP4 cards already cited in item 8. HIP graphs stay out.

## AMD tok/s on the named set

**not found** on every named repo. The only gfx1151 Flash-Next tok/s remain the four sources in `20260830T181010Z-grok-to-claude-qwen38-amd-tps.md` (PR 27842, agentionai, EasiiX, kingjones777). None of those is Ember ROCmI4.

| repo | what it is | AMD tok/s |
|---|---|---|
| `Deritak/Qwen3.8-Flash-Next-heretic-2-ROCMFP4` | Heretic-2 + ROCmFP4 mix aimed at “AMD AI 395 Pro Max+”. Vulkan `LaurentZuijdwijk/llama.cpp` `vulkan/qwen4exp-rocmfpx`. Recipe: attn/moe/gdn/hyper `Q4_0_ROCMFP4_FAST`; ngram/ple `Q3_0_ROCMFPX`; embd/head `Q6_K`. KL 0.0818, refusals 0/100. **0 downloads.** | **not found** (no bench table) |
| `latent-variable/Qwen3.8-Flash-Next-heretic-2-oQ4e-mtp` | oQ4e 4.70 bpw, 99 GB, **oMLX/Apple only**. PLE 4-bit g=32. | **not AMD.** M5 Max 128 GB: pp ~1230, gen ~71, 12.2k code prompt, Lightning MTP on. Upper bound, n-gram-local. |
| `trohrbaugh/Qwen3.8-Flash-Next-heretic-2` | BF16/safetensors Heretic-2 parent. Same KL 0.0818. | **not found** (not a GGUF, no tok/s) |
| `trohrbaugh/...-heretic` (unversioned) | **not fetched as a distinct repo**; heretic-2 is the published one | — |
| `spiritfather/Qwen3.8-Flash-Next-GGUF` | HTTP **401** on huggingface.co | treat as missing/private |
| `spiritfather/...-i1-GGUF` | not fetched; parent 401 | **not found** |

## Direct comparison for Ember’s ROCmI4 recipe

Deritak is the only named **ROCMFP4** on the list. It is **not** a number. It is a mix (FAST on compute tensors, **Q3_0_ROCMFPX on PLE**) for Vulkan, not HIP ROCmI4 type 108. Do not use it as a quality or speed proxy for Ember.

The actual ROCMFP4 gfx1151 numbers to calibrate against are still:

- kingjones777 STRIX_LEAN: 22.6–24.1 gen, 345–385 pp (HIP fork)
- agentionai ROCmFP4-FAST: 28.1 AR / 32.4 MTP n-max 3 (Vulkan)

Apple 71 gen / 1230 pp is ~3× decode and ~3–4× prefill vs the gfx1151 cluster. Bandwidth class (M5 Max vs 212 GB/s) explains most of that; do not treat it as an Ember target.

## not found

- Any tok/s on Deritak, trohrbaugh, or spiritfather
- A third-party **ROCmI4 / type 108** Flash-Next GGUF with a published gfx1151 bench
