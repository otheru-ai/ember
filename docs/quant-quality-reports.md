# Quantization quality reports

Ember's quant report pipeline produces a human-readable Markdown report,
machine-readable JSON/CSV data, and dependency-free SVG plots. Its main goal is
to distinguish three effects that are often collapsed into one benchmark:

1. **Upstream → quantization source:** model transformations, engine differences, and
   chat-template differences. Abliteration belongs here.
2. **Quantization source → quant:** the quantization-only estimate, measured with the
   same engine, tokenizer, template, and frozen continuation corpus.
3. **End-to-end behavior:** tool calls, identifiers, repetition, format safety,
   latency, memory, and throughput under the serving configuration users run.

A report never calls a quant "lossless" from perplexity alone. Average logit
metrics can hide a rare but important tool-call flip, while sampled answers are
too noisy to replace token-probability measurements.

## Inputs

`scripts/quant_quality_report.py` accepts these independent evidence layers:

| Input | Purpose |
|---|---|
| `--reference-scores` + `--quant-scores` | Dwarfstar `score_official` TSVs for the quantization source and quant. These provide target-token NLL, greedy-prefix agreement, and agreement with frozen hosted-API top-logprob slices. |
| `--distribution` | Optional `llama-perplexity --kl-divergence` log or normalized JSON. This adds full-vocabulary KLD, same-top-token probability, and probability-delta statistics. |
| `--behavior` | JSONL from `quant_behavior_eval.py`, including agent/tool validity, identifier integrity, visible reasoning/DSML leakage, and repetition. |
| `--runtime` | JSON with artifact size, peak RSS, prefill throughput, and decode throughput for `reference` and `quant`. |
| `--structural-audit` | JSON confirming GGUF bounds, tensor counts/types, and absence of trailing or truncated data. |
| `--tensor-error` | Optional TSV of sampled reconstruction error by tensor class. |
| `--variants` | Optional JSON/TSV leaderboard for Unsloth-style size-vs-KLD, same-top-token, and behavioral frontier plots. |
| `--modes` | Optional JSON/TSV comparison of serving modes, separating quantization effects from prefill and speculative-decoding effects. |
| `--metadata` | Identity, exact provenance, limitations, release gates, and notes. |

All supplied input files are SHA-256 hashed into `report.json`. Artifact hashes
should also be placed in metadata; artifact hashing is deliberately not implicit
because a DeepSeek V4 source file can exceed 150 GB.

## Frozen continuation scoring

Dwarfstar carries the local `score_official` scorer and an example frozen
continuation fixture. Audit the response model, revision, provider, and
logprob coverage before using a fixture; recollect it when evaluating a newer
checkpoint such as 0731. Run the scorer twice with the same commit and settings:

```bash
make -C /path/to/ds4 gguf-tools/quality-testing/score_official

/path/to/ds4/gguf-tools/quality-testing/score_official \
  /models/quantization-source.gguf \
  /path/to/ds4/gguf-tools/quality-testing/data/flash/manifest.tsv \
  /reports/source.tsv 4096

/path/to/ds4/gguf-tools/quality-testing/score_official \
  /models/candidate-quant.gguf \
  /path/to/ds4/gguf-tools/quality-testing/data/flash/manifest.tsv \
  /reports/quant.tsv 4096
```

Do not compare scores produced with different prompt templates, tokenizer
metadata, expert counts, or scorer commits. The per-case TSV is retained so an
aggregate improvement cannot hide a badly regressed prompt.
If the scorer cannot decode a custom quant type, omit the paired score inputs
and use full-logit comparison in an engine that can decode both artifacts. Do
not compare NLL values emitted by different engines as if they were a
quantization-only delta.

The same frozen set can be rendered deterministically for a matched-engine
`llama-perplexity`/KLD run:

```bash
python3 scripts/quant_manifest_corpus.py \
  --manifest /path/to/data/flash-0731/manifest.tsv \
  --shuffle-seed 7301 --out /reports/flash-0731-heldout.txt
```

The optional deterministic shuffle prevents a short chunk-limited run from
sampling only adjacent manifest cases. The command prints the corpus size,
case count, shuffle seed, and SHA-256 hash. Retain those
with the exact context/chunk arguments and the saved source-logit file.

## Behavioral replay

The checked-in smoke suite covers knowledge, reasoning, code, structured
output, tool selection/restraint, exact identifiers, multilingual output,
repetition, and raw protocol leakage. Run the same cases against the source and
candidate serving configurations:

```bash
python3 scripts/quant_behavior_eval.py \
  --cases share/quant_eval/agentic_cases.jsonl \
  --endpoint http://127.0.0.1:8090/v1/chat/completions \
  --model deepseek-v4-flash --variant reference \
  --request-overrides share/quant_eval/ember_reasoning_none.json \
  --out /reports/behavior.jsonl

python3 scripts/quant_behavior_eval.py \
  --cases share/quant_eval/agentic_cases.jsonl \
  --endpoint http://127.0.0.1:8090/v1/chat/completions \
  --model deepseek-v4-flash --variant quant \
  --request-overrides share/quant_eval/ember_reasoning_none.json \
  --out /reports/behavior.jsonl --append
```

For hosted baselines, pin the provider rather than accepting an arbitrary
router fallback. Put provider-specific request fields in a JSON object and
pass it with `--request-overrides`; the exact file is then part of the retained
evaluation inputs. Equivalent controls can use different wire formats: the
checked-in OpenRouter override uses `reasoning.effort=none`, while Ember's Chat
Completions endpoint uses `reasoning_effort=none`. Never compare a hosted
thinking-off run to Ember's default thinking-on mode under the same short
`max_tokens` cap. `--temperature` defaults to zero for deterministic replay.

The runner stores raw responses next to the JSONL and records their hashes.
Add project-specific long-horizon replays before a release; the smoke
suite is not a substitute for a representative multi-turn agent corpus.

## Runtime and tensor inputs

Runtime JSON uses byte counts and tokens/second:

```json
{
  "reference": {
    "artifact_bytes": 161900196384,
    "peak_rss_bytes": null,
    "prefill_tokens_per_second": null,
    "decode_tokens_per_second": null
  },
  "quant": {
    "artifact_bytes": null,
    "peak_rss_bytes": null,
    "prefill_tokens_per_second": null,
    "decode_tokens_per_second": null
  }
}
```

Tensor-error TSV requires `tensor`, `class`, and any of `nrmse`,
`weighted_nrmse`, or `cosine`. Keep the sampling seed and method in metadata.
Weighted NRMSE should use an independent activation corpus, not the importance
matrix calibration samples, or it can reward calibration overfitting.

Generate a deterministic, stratified sample with the exact ggml library that
implements the candidate's custom quant type:

```bash
python3 scripts/gguf_tensor_error.py \
  --library /path/to/libggml.so \
  --reference /models/source.gguf \
  --candidate /models/quant.gguf \
  --imatrix /models/held-out-imatrix.dat \
  --rows-per-tensor 32 --tensors-per-class 4 \
  --out /reports/tensor-error.tsv
```

If the same matrix calibrated the quant, label weighted NRMSE as in-sample and
also run against a held-out matrix. Unweighted NRMSE and cosine remain useful
sanity checks, but neither substitutes for output-distribution evaluation.

## Generate the report

Copy and fill `share/quant_eval/report_metadata.example.json`, then run:

```bash
python3 scripts/quant_quality_report.py \
  --metadata /reports/metadata.json \
  --reference-scores /reports/source.tsv \
  --quant-scores /reports/quant.tsv \
  --distribution /reports/llama-kld.log \
  --behavior /reports/behavior.jsonl \
  --runtime /reports/runtime.json \
  --structural-audit /reports/gguf-audit.json \
  --tensor-error /reports/tensor-error.tsv \
  --variants /reports/variants.tsv \
  --modes /reports/modes.tsv \
  --out-dir /reports/deepseek-v4-flash-0731-rocmfp2
```

The output directory contains:

- `report.md`, with a plain-language degradation statement and embedded plots;
- `report.json`, the complete computed record and provenance;
- `case_metrics.csv`, retaining every continuation delta;
- `behavior_cases.csv`, retaining every behavioral classification, raw-response
  path, and response hash;
- SVG plots for per-case NLL, upstream agreement, behavior, runtime, and tensor
  reconstruction when their corresponding inputs are present, including
  artifact/peak-memory footprint and throughput, plus labeled
size/quality frontier plots when two or more variants are supplied.
  A serving-mode safety plot is also emitted when `--modes` is supplied.

The optional variants TSV starts with `name` and may include `artifact_bytes`
or `size_gib`, `bpw`, `perplexity`, `mean_kld`, `p99_9_kld`, `maximum_kld`,
`same_top_probability_pct`, `behavioral_pass_rate`, and throughput fields.
Use decimal fractions for `behavioral_pass_rate` and percentages (0–100) for
`same_top_probability_pct`. Every row must come from the same evaluation setup.

The optional modes TSV starts with `name` and may include `artifact`, `prefill`,
`drafter`, `tool_result_guard`, `cases`, `behavioral_pass_rate`,
`tool_result_pass_rate`, `repetition_detected_rate`,
`identifier_integrity_rate`, and prefill/decode
throughput. Rates are decimal fractions. Use it when the same quant behaves
differently under exact versus approximate prefill or with a drafter loaded;
those are serving-path changes, not quantization-only degradation.

The command exits nonzero when a configured gate fails or cannot be evaluated.
Thresholds are intentionally explicit metadata, not hidden universal defaults.
For a release, gates should cover both quant-only output divergence and agentic
failure modes. A report with no gates is labeled **NOT GATED**.

## Reading the result

- **NLL/perplexity delta** is the most broadly available quant-only signal.
  Lower is better; `exp(NLL delta) - 1` is the relative perplexity change.
- **KLD and same-top-token probability** compare complete output
  distributions and are preferred when the runtime can dump full logits.
  Report mean, 99.9th-percentile, and maximum KLD when available: mean KLD can
  conceal rare but catastrophic token-distribution outliers.
- **Official top-N recall and pair ordering** use only the hosted API's exposed
  top-logprob slice; they are useful but are not full-vocabulary KLD.
- **Greedy-prefix agreement** is intentionally brittle. A one-token logit flip
  can cause a different valid continuation, so inspect per-case plots and raw
  behavior instead of treating it as a standalone quality score.
- **Behavioral replay** communicates user-visible risk: malformed tool calls,
  repeated loops, changed identifiers, and protocol leakage.

Always publish corpus identity, prompt template, engine commits, sampling
settings, context length, expert count, importance-matrix provenance, and all
artifact hashes alongside the plots.

## Improving the next recipe

Treat a mixed-precision recipe as a budgeted search, not a single global type.
Start with a fixed artifact-size/BPW budget, then use held-out activation error
and output-distribution tails to promote only the most sensitive tensor groups.
For DeepSeek V4 this means evaluating routed down, gate, and up experts
separately by layer while continuing to protect attention, shared experts,
embeddings, output, indexer, compressor, and hyper-connection tensors. Compare
every candidate on the same frozen corpus before choosing the size/quality
frontier.

Keep calibration and evaluation data disjoint. Calibration should be diverse
enough to exercise long context, tool protocols, multilingual text, code, and
the model's routed experts; a second importance matrix from held-out prompts is
useful for detecting calibration overfit. Average reconstruction error is only
a published report should favor matched-engine NLL/KLD,
99.9th-percentile divergence, same-top-token probability, and user-visible
behavioral flips.

Relevant primary and project references:

- [DeepSeek V4 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [llama.cpp perplexity and KL-divergence tooling](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)
- [llama.cpp importance-matrix discussion](https://github.com/ggml-org/llama.cpp/discussions/5263)
- [Dwarfstar's DeepSeek V4 implementation and scorer](https://github.com/antirez/ds4)
- [Antirez DeepSeek V4 GGUF recipe](https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/README.md)
- [Unsloth Dynamic 2.0 GGUF method](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- [Unsloth's size/quality benchmark presentation](https://unsloth.ai/docs/models/gguf-benchmarks)
