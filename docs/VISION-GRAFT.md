# DeepSeek-V4-Flash-Vision language seam

This branch (`vision/mtmd-graft`) is the staged Ember port for
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. Planning, upstream pins and the
reference implementation live in the sibling `ds4-vision` repository.

The current stage implements the language-model side of the official contract.
It does not yet claim a native vision tower or a production-ready multimodal
server.

## Learned-marker contract

The model does not use the old DS4IMGE1 routing palette. Each image is expanded
into an official N-layout made from five vocabulary-relative sentinel IDs:
`IMAGE_START`, `IMAGE_PAD`, `IMAGE`, `IMAGE_NEWLINE`, and `IMAGE_END`. The four
non-image rows come from learned projector marker weights; `IMAGE` rows come
from the aligned vision-tower output.

Ember keeps two inputs separate at the prefill seam:

- the vocabulary embedder receives a bounds-safe copy with sentinel IDs
  replaced by token zero, then every covered row is overwritten by its exact
  request-owned learned embedding;
- the graph receives the original sentinel-bearing IDs for image visibility
  and MoE routing.

Every run is validated before embedding or graph construction. Runs must be
ordered, non-overlapping, finite, width-correct, and exactly match the expanded
prompt. A prefill chunk, restore boundary, or snapshot boundary may not bisect a
run. Chunk sizing derives from the computed block length: although the LM-grid
budget is 384, legal alignment padding can make the complete emitted block 405
tokens long.

## Routing and attention

The first three hash-routed MoE layers retain hash routing for text but bypass
the token hash for image sentinels and score-route them with the per-layer
`exp_probs_b_vl.bias`. Later layers use the ordinary text bias for text rows and
the vision bias for sentinel rows. A missing or partial vision-bias set fails
closed for image requests and remains valid for text-only checkpoints.

Layer-major prefill widens raw sliding-window attention only within a complete
learned image block. Compressed rows remain causal. Image-bearing chunks do not
reuse cached graphs because the cache key does not carry their row partitions
or visibility counts.

## Current execution boundary

The offline language-seam gate injects prepared embeddings as request-owned
runs. Ember deliberately does not load one global artifact while ignoring the
request image bytes; doing so would make different images resolve to the same
content while appearing successful.

Until the continuous-batch scheduler can reserve an indivisible image-block
quantum, resident image sessions fail with
`vision_resident_prefill_unsupported`. The ordinary nonresident and
snapshot-restore paths accept complete runs. DSpark is disabled for image
requests in this stage.

Image tool turns intentionally skip post-tool snapshots until the durable
cache format carries media identity. Their physical KV contains the full image
frontier and cannot safely be published under a shorter pre-image key. This is
a known cold-TTFT tradeoff, not evidence about tower performance.

## Remaining work

The next product tranche must implement the model's own 32-block ViT and
pixel-shuffle aligner, bind its output to the actual request bytes, and pass the
frozen image-grounding corpus. Resident scheduling then needs an explicit
indivisible minimum quantum rather than truncating or stalling a learned block.
Only after those correctness gates may hardware throughput be measured.
