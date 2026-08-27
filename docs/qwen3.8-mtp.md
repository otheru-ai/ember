# Qwen3.8-Flash-Next MTP contract and strict replay

Qwen3.8-Flash-Next carries one embedded multi-token-prediction block. Ember can
extract it losslessly, convert and load a matching quantized GGUF companion,
run its input fusion, full-QSA block and MoE graph with independent draft state,
and reconcile partial target verification without retaining rejected target
state. This is a runnable correctness-first path, not yet a throughput claim:
target verification is still serial q=1 rather than a batched verify graph.

## Audited upstream contract

The source model is pinned at
`f5d08274bafd880402bd16f5e3e6c514136ec06c`. Its text config declares one
hybrid MTP layer whose only layer is full attention, with shared main token
embeddings and output head. The 31 MTP tensors contain 2,607,150,848 BF16
parameters (4.856 GiB).

There is no upstream Qwen4Exp execution graph to import:

- Transformers `36deb0b53ed0863f4b4dfdea23dcaec7f3df3701` explicitly lists
  `^mtp.*` under `_keys_to_ignore_on_load_unexpected`.
- llama.cpp PR #27742 and its rotated-KV follow-up #27774 set the Qwen4Exp
  converter's `supports_mtp_export = False` and `no_mtp = True`.
- The Qwen3.5 MTP implementation is useful scheduling prior art, but it cannot
  be copied as the compute contract. Qwen4Exp's draft block has four-stream
  gated residuals and QSA with an index cache.

## Streaming export

The exporter reads only safetensors headers plus one bounded copy buffer. It
validates the exact pinned config, tensor set, BF16 dtype, shapes, byte extents,
and safe shard names before creating anything. It then writes an atomic
MTP-only safetensors file and a provenance manifest:

```bash
python3 scripts/qwen_mtp_export.py \
  --source-dir /models/Qwen3.8-Flash-Next/snapshots/f5d08274bafd880402bd16f5e3e6c514136ec06c \
  --output /scratch/qwen3.8-mtp/Qwen3.8-Flash-Next-MTP-BF16.safetensors \
  --manifest-out /scratch/qwen3.8-mtp/export.json \
  --matrix-quant Q4_0_ROCMI4
```

`--matrix-quant` records the required matching-main conversion contract; the
safetensors export remains lossless BF16. It does not silently run a different
quantizer. `--format gguf` writes the exact loadable companion schema, and
`--quantizer ... --quantized-output ...` produces the ROCMI4 serving artifact.
The companion shares `token_embd.weight` and `output.weight` from the target;
norms remain BF16. Fusion projections and router precision remain release-recipe
decisions and must be evaluated for acceptance rate rather than guessed.

Use `--print-contract` to inspect the exact tensor list without model files.
Existing output is never replaced unless `--force` is explicit.

## Runtime state boundary

The trained draft consumes the target pre-output-mixer state `h_p` with 10,240
values and the next token embedding `x_(p+1)` with 2,560 values. The draft must
own a separate QSA K/V/index cache. It must not borrow the target cache or
mutate the target's GDN recurrence, width-four causal-convolution history, PLE
token/convolution history, HC streams, or M-RoPE positions.

`qwen4exp_replay_accepted_prefix()` implements the strict target boundary from
HaloSpecKV `60ff854bdc25e27ee211ac0c4df896e9379edd3f`:

1. checkpoint the complete target state before all bounded draft rows;
2. verify the candidates;
3. retain the verified state only if every draft row is accepted;
4. otherwise restore the complete checkpoint and run each accepted input row
   through the ordinary target q=1 step;
5. if a replay step fails, return to the original checkpoint transactionally.

The GPU-free test mutates and compares every state family, including COW QSA
slabs and shared GDN vectors. This prevents a scheduler implementation from
testing only token position while leaving rejected recurrent state alive.

At runtime, set `DFLASH_QWEN_MTP` to the quantized companion GGUF and optionally
set `DFLASH_QWEN_MTP_DEPTH` to an integer from 1 through 4. The backend keeps
the draft QSA cache and target-HC frontier synchronized through prompt ingest,
generation, snapshot save/restore and reset. It records proposal/acceptance
telemetry and falls back to ordinary autoregressive decode when MTP is disabled
or the request requires it.

## Remaining performance-only work

The following work is required before `--spec-type` or an MTP speed result is
honest:

1. replace serial q=1 candidate verification with a bounded batched target
   verifier that invokes strict replay on every partial acceptance;
2. differential-test greedy AR against MTP at short and long contexts, after
   snapshot restore, and across partial rejection;
3. measure unprofiled end-to-end throughput and acceptance on exclusive
   gfx1151 before profiling the target verify and draft kernels separately.

Until those gates pass, MTP remains opt-in and Ember's release baseline remains
the token-exact autoregressive path.
