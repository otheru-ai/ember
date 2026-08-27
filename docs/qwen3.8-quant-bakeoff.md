# Qwen3.8 Flash Next quant and intervention bakeoff

This is an experiment protocol, not a selected release recipe. A recipe is
confirmed only after the scripts accept measured correctness, quality,
throughput, artifact-size, RSS, GTT, and accounted UMA evidence from `otheru`.
Estimated values and missing arms fail closed.

## Builder limits and bounded conversion

The certification host reports `MemTotal: 131150288 kB`, exactly
134,297,894,912 bytes (about 125.07 GiB). Its root filesystem has about 1.487
TiB free; `/srv/models` has only about 295.4 GiB free. The pinned source weight
inventory is 360,000,192,888 bytes, so both the snapshot and conversion
workspace belong on the root filesystem.

The live gfx1151 mapping ceiling is slightly smaller than physical memory:
`/sys/module/ttm/parameters/pages_limit` is 32,505,856 pages, exactly
133,143,986,176 bytes (124.0 GiB), and `rocminfo` reports the same 124.0 GiB
global pool. Certification therefore rejects a measured GTT peak above that
cap even when total UMA remains below `MemTotal`.

Ordinary conversion is not viable on this host: its approximately 204 GB
intermediate representation exceeds RAM. The bounded path uses the pinned
llama.cpp converter's `--use-temp-file` mode. That mode cannot split while
converting, so Ember performs this private lifecycle:

1. stream safetensors into one private BF16 GGUF, spilling through `TMPDIR`
   inside the transaction directory;
2. use `llama-gguf-split` from the same pinned llama.cpp checkout to create 48G
   shards;
3. verify the complete ordered GGUF set and remove the unsplit BF16 file;
4. preflight and quantize the split set with `--keep-split`;
5. atomically commit the verified directory without overwriting an existing
   result.

The input snapshot, temporary payload, unsplit output, and split output can
coexist at different points, so the normal 1 TiB free-space gate remains. The
120 GiB physical-RAM floor merely admits the mechanically bounded route; it is
not evidence that conversion succeeded within memory. The build record leaves
peak RSS and wall time pending until the target run measures them.

```sh
python3 scripts/qwen_quantize.py \
  --snapshot-dir /root/qwen-work/snapshot \
  --snapshot-revision f5d08274bafd880402bd16f5e3e6c514136ec06c \
  --stock-control \
  --llama-cpp-dir /root/qwen-work/llama.cpp \
  --gguf-splitter /root/qwen-work/llama.cpp/build/bin/llama-gguf-split \
  --bounded-memory-temp \
  --rocmfpx-dir /root/qwen-work/ROCmFPX \
  --ember-dir /root/qwen-work/ember \
  --ember-revision "$(git rev-parse HEAD)" \
  --quantizer /root/qwen-work/ember/build-rocm/ember-gguf-quantize \
  --work-dir /root/qwen-work/stock-rocmi4 \
  --execute
```

The stock control carries no `ember.intervention.*` metadata and is always
final-ineligible. It exists to establish correctness/quality/performance and
to capture the 48×2560 per-prompt mixed-input activations. A release package
still requires a separately measured intervention artifact.

## Pinned and disjoint corpora

The adapter accepts only OtherU quant-pipeline revision
`a3c6a728510f91394e991504951ac316cd3a89af` on branch
`ember-contract-and-drafter-fix`, and verifies each raw input digest before
reading it.

```sh
python3 scripts/qwen_corpus_adapter.py \
  --source-dir /root/qwen-work/otheru-quant-pipeline \
  --output-dir /root/qwen-work/qwen-corpora
```

At the pinned revision the deterministic result is:

- 32 harmless and 32 harmful direction-extraction rows;
- 134 sweep-validation rows;
- 134 final-heldout rows;
- zero canonical-message overlap among all four outputs.

Within each evaluation suite, rows are sorted by SHA-256 of canonical chat
messages and divided equally. The final half cannot select a recipe. The
upstream `overtrigger.json` is digest-pinned but deliberately excluded because
it contains environment-derived private material; a separately reviewed,
sanitized tool-use suite is required before tool-use quality can pass.

## Bakeoff and measured decision

Generate the expanded 4-lambda × 4-layer-policy plan:

```sh
python3 scripts/qwen_bakeoff.py \
  --corpus-dir /root/qwen-work/qwen-corpora \
  --output /root/qwen-work/qwen-bakeoff-plan.json
```

The sweep uses positive projection-removal strengths 0.25, 0.5, 0.75, and
1.0 over all 48 layers, upper 24, upper 12, and the 36 non-QSA layers. It first
selects an intervention configuration using only sweep-validation. That fixed
configuration is then compared as ROCMI4 exact-dequant, ROCmFP4 FAST
exact-dequant, and ROCMI4 W4A4. W4A4 and the stock model are performance
controls and are final-ineligible. Exactly one already-selected eligible arm
is evaluated on final-heldout.

The ROCmFP4 FAST post-encoding audit now dispatches through the actual stored
destination type and its cross-decoder GPU-free regression passes. The arm is
still unpromoted until its exact intervened artifact passes the real-weight
gfx1151 differential, quality, memory, and performance gates below.

After the digest-matched stock ROCMI4 control and MTP companion exist, capture
directions with `scripts/qwen_capture_control.py`. The operator supplies every
image, build-record, shard, MTP, recipe, contract, and extraction-corpus digest
explicitly. The driver hashes large model artifacts with direct I/O, takes the
fixed GPU lock, stops and masks production, captures exactly 32 good plus 32
bad final-row activation records, and restores production before generating
the deterministic 4×4 intervention grid. It never opens the sweep or final
JSONL files and never quantizes or publishes anything. Use `--dry-run` first;
the printed plan is side-effect free. On the certification host the pinned
corpus directory is `/srv/ember/qwen3.8-otheru-corpus-a3c6a728`.

After every required row is measured, adjudicate it with:

```sh
python3 scripts/qwen_bakeoff.py \
  --plan /root/qwen-work/qwen-bakeoff-plan.json \
  --results /root/qwen-work/qwen-bakeoff-results.json \
  --output /root/qwen-work/qwen-bakeoff-decision.json
```

Every row must include at least three prefill and decode samples, audited
quality and differential results, the exact main/MTP/mmproj artifact byte
inventory, enabled companions, the exact 134,297,894,912-byte host MemTotal,
and measured peak RSS, GTT, and deduplicated accounted UMA bytes from the
runner sampler, including the exact live TTM `pages_limit`. The hard medians
are 412 tok/s prefill and 39.49 tok/s decode.
Both the static artifact-plus-reserve-plus-companion sum and measured UMA peak
must fit real host MemTotal, which is stricter than the nominal 128 GiB
architectural budget. The decision remains non-publishing; release packaging
and hardware certification are separate gates.
