# Can Ember load a third-party Qwen3.8-Flash-Next GGUF?

Assessed without downloading any of them. A GGUF's header — magic, KV metadata
and the full tensor table — sits at the front of the file, so an HTTP range
request for the first 80 MB answers the question in a couple of minutes and a
rounding error of bandwidth. The alternative was ~87 GiB per candidate.

Ember's gate is `qwen4exp_weight_type_supported`
(`engine/dflash/qwen4exp/qwen4exp_state.cpp:24-32`):

    any tensor      F32, F16, BF16
    matrices also   Q8_0, Q6_K, Q3_0_ROCMFPX, Q4_0_ROCMI4, Q4_0_ROCMFP4_FAST

plus a hard structural requirement: a **single** `per_layer_token_embd.weight`
for the PLE table (`qwen4exp_loader.cpp:410-414`, enforced at `:448`).

## Results

| candidate | arch | blocker |
|---|---|---|
| `agentionai/…-FAST-imatrix-GGUF` **`v2/`** | `qwen4exp` | **96 Q4_K tensors** — nothing else |
| `agentionai/…-FAST-imatrix-GGUF` root (`-ple16`) | `qwen4exp` | same Q4_K, **plus** a split PLE |
| `jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4` | `qwen4exp` | **520 tensors** across three unsupported types, **plus** an out-of-GGUF PLE |

### agentionai `v2/` — one blocker, and it is Ember's own allow-list

1224 tensors. Metadata matches Ember's hardcoded expectations exactly:
`block_count 48`, `embedding_length 2560`, `expert_count 512`,
`expert_used_count 10`, `embedding_length_per_layer_input 160`.

    type   0 F32               388     accepted
    type   1 F16                 1     accepted
    type   8 Q8_0              144     accepted
    type  12 Q4_K               96     REJECTED
    type  14 Q6_K              520     accepted
    type  30 BF16               24     accepted
    type 101 Q4_0_ROCMFP4_FAST  50     accepted
    type 104 Q3_0_ROCMFPX        1     accepted  (the PLE table)

`per_layer_token_embd.weight` is present as a single tensor, and the loader
already handles this model's **split** `ffn_gate_exps` / `ffn_up_exps` form as
well as the fused one (`qwen4exp_loader.cpp:417-435`).

The 96 Q4_K tensors are exactly `blk.N.ffn_gate_exps.weight` and
`blk.N.ffn_up_exps.weight` — 48 layers times two. **This is Ember's allow-list,
not a backend limitation**, and that is verified rather than assumed:

- `supports_op` for `MUL_MAT_ID` has **no type whitelist**. The grouped-src
  branch gates on `ggml_is_quantized(a->type)` plus shape and buffer conditions
  (`ggml-cuda.cu:5252-5271`); the rest is generic acceptance with specific
  exclusions (MUSA Q2_K at `:5297`, split buffers, F16-b-with-non-F16-a). Q4_K
  trips none.
- MMQ implements Q4_K: `ggml_cuda_should_use_mmq` lists it (`mmq.cu:443`),
  with a dispatch case (`:99-100`), DP4A tile sizes (`mmq.cuh:258`), an MMA
  tile layout (`:315`), and a Q4_K/Q5_K `ne11 <= 256` case (`mmq.cu:501`).

A Q4_K expert tensor therefore reaches MMQ by the same path Q6_K already does,
and Ember's ROCmFPX-specific MoE routes select among supported paths rather
than narrowing the type set.

### agentionai root (`-ple16`) — additionally incompatible

1239 tensors, same Q4_K blocker, and `per_layer_token_embd.weight` is **absent**.
The PLE ships as **16** `ple_ngram_embd.N.weight` tensors (type 104), one per
PLE head. Ember's loader requires the single tensor and fails at `:448`.

Worth noting the split is per-head and Ember's runtime already consumes PLE
per-head — `qwen4exp_ple_rows` returns 16 row indices and `run_ple` reads 160
floats per head. So supporting it is a loader change, not an algorithmic one.
Use `v2/` regardless; this is recorded only so the difference is not
rediscovered.

### jcbtc CIRU-STRIX-IU4 — a worse fit than the name suggests

**Contains no type 108 at all**, despite "IU4". 1223 tensors:

    type  0 F32    388    accepted
    type  3 Q4_1   144    REJECTED   (the experts)
    type  7 Q5_1    48    REJECTED
    type  8 Q8_0   290    accepted
    type 13 Q5_K   328    REJECTED
    type 30 BF16    25    accepted

520 rejected tensors across three types. And `per_layer_token_embd.weight` is
absent entirely — the PLE ships **outside the GGUF** as `ple/ple.payload.bin`,
`ple/ple.scale.bf16` and `ple/ple.manifest.json`, a bespoke sidecar Ember has no
reader for.

Loading it would mean three new quant types plus a new PLE ingest path. Not
recommended.

## Recommendation

`agentionai/…-FAST-imatrix-GGUF` **`v2/`** is the only near-miss, and the gap is
one allow-list entry. See the Q4_K spec in `.coord/msg/` (claude 333).

## Method note

Range-fetching a GGUF header is cheap enough to be routine, and should be the
first step for any third-party checkpoint. Check the magic before trusting
anything else — a quantizer can emit a full-size file with a zeroed header, and
size alone will not tell you.
