# Vision graft — why this fork exists

This branch (`vision/mtmd-graft`) adds image input to Ember so DeepSeek-V4-Flash
can attend over pixels instead of over a caption.

Planning, upstream pins, staged de-risking and the inspection tooling live in the
sibling repo **`ds4-vision`**. Read `ds4-vision/docs/PLAN.md` before starting —
in particular, **stages 2 and 3 must pass before any work happens here.** They
test whether the trained projector still aligns after our abliteration and
2.5-bit expert quantization, and neither needs a line of Ember code.

## What has to be built here

1. **Port `libmtmd` + `clip.cpp` + `kimik3.cpp`** from llama.cpp. Ember has no
   multimodal path today — verified: zero `mmproj` / `mtmd` / `clip_` /
   `llava` / `image_embed` symbols in the shipped binary.
2. **PatchMerger projector** — 6 tensors:
   `pre_norm[1152] -> 2x2 merge -> Linear(4608,4608) -> GELU -> Linear(4608,4096)`.
3. **Routing bridge** — the DS4-specific piece with no analogue in any existing
   mtmd model. Image positions cycle a fixed 64-entry palette of *hash-routing*
   IDs (0..121562, well beyond the 256 experts/layer, so they must go through the
   same hashing path text tokens use). Text routing IDs are preserved exactly.
   The palette is part of the trained contract — do not improvise it.
4. **Image-token splicing** into the embedding sequence.

## Two things that will bite

**MoonViT's head dim is not `n_embd / n_head`** — upstream added
`clip_hparams::n_embd_head` for exactly this. A port that assumes the usual
relation gets wrong attention shapes. The tower is 27 blocks at `n_embd = 1152`
with fused `wqkv [3456, 1152]`.

**Ember currently accepts image parts and silently ignores them.** Posting an
`image_url` returns a normal completion rather than an error, so a harness can
believe vision works. Mainline llama.cpp returns
`HTTP 500: image input is not supported`. Whatever lands here, make the
unsupported path *loud* — the silent drop is worse than the missing feature.

## Precedent for the port

`otheru-quant-pipeline/scripts/14-port-rocmfpx-onto-master.sh` ports ggml types
onto a llama.cpp base and documents the three registration points a new type
needs. Same shape of work, larger surface.

## Fallback

Mainline llama.cpp already runs MoonViT. A DS4 GGUF served by patched mainline
llama.cpp is a fallback if this port stalls — at the cost of the ROCMFPx types
and DSpark.
