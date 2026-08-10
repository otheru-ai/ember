# Vision graft — why this fork exists

This branch (`vision/mtmd-graft`) adds image input to Ember so DeepSeek-V4-Flash
can attend over pixels instead of over a caption.

Planning, upstream pins, staged de-risking and the reference implementation live
in the sibling repo **`ds4-vision`**. Read `ds4-vision/docs/PLAN.md` first.

> **Revised 2026-08-10.** An earlier version of this document repeated two claims
> that reading the pinned upstream sources disproved. Both are corrected below;
> they are called out rather than quietly edited because either would have sent a
> port down the wrong path.

## What changed, and why the port got smaller

The original plan here was: port `libmtmd` + `clip.cpp` + `kimik3.cpp`, add a
projector type, build a routing bridge, splice image tokens. That is still the
shape of a *product*. It is **not** what validation requires, because of two
findings.

### 1. Ember already separates hidden states from routing IDs

`engine/dflash/deepseek4/deepseek4_backend.cpp:1176` — the prefill loop:

```cpp
std::vector<float> embed(w_.n_embd * n_tok);
w_.embedder.embed(tokens.data() + i, n_tok, embed.data());
```

`CpuEmbedder::embed` (`engine/dflash/internal.h:144`) dequantizes token rows into
a plain contiguous **f32** buffer with `n_embd` as the fast axis. The graph is
then handed `embed` and `token_ids` as **separate arguments**, hash routing
reading `token_ids` and the residual stream reading `embed`.

That is exactly the seam the graft needs:

- the **routing bridge needs no graph change** — put palette IDs in `tokens`;
- **image embeddings splice in by overwriting rows of `embed`**, a memcpy per
  token, immediately after that one call;
- **`clip.cpp` / `libmtmd` are not on the critical path for validation** — the
  vision tower runs in Python today and is cross-checked against Moonshot's own
  implementation at 7e-6 relative error (`ds4-vision/tools/crosscheck_tower.py`).

### 2. The tower is Kimi-**K2.6**, and `kimik3` is the wrong reference

The upstream package implements neither tower nor projector; it imports both from
SGLang's `kimi_k25`, pinned by `deepseek_vision.sglang_source_commit`. llama.cpp's
`kimik3.cpp` targets K3's MoonViT-V2 — related, not the same network.

**Correction:** the earlier claim that "MoonViT's head dim is not `n_embd/n_head`"
is true of K3 and **false of ours**. Our tower is 27 blocks, `n_embd = 1152`,
16 heads, so head dim is **72 = 1152/16** — the ordinary case. `n_embd_head`
exists upstream for K3's benefit and does not apply here.

**Correction:** the earlier claim that the palette holds "*hash-routing* IDs …
well beyond the 256 experts/layer" is wrong. They are **token IDs**. Upstream's
`build_sglang_routing_ids` substitutes them into `input_ids`; DS4's ordinary hash
routing then runs on them. Verified against our own artifact
(`ds4-vision/tools/check_routing_tables.py`): `blk.N.ffn_gate_tid2eid.weight` is
`[6, 129280]`, `token_embd.weight` is `[4096, 129280]`, and the palette spans
0..121562 — in range for both.

The placeholder ID is **129280**, which *equals* `vocab_size` and is therefore out
of range for both tables. Substitution is not a tuning choice; an unsubstituted
placeholder is an out-of-bounds read.

## The validation path (stage 2b) — build this first

Goal: prove DS4 produces image-grounded text from real projector output, before
spending effort on a product-quality multimodal path.

**Input.** `ds4-vision/tools/make_image_embed.py` writes a self-describing
sidecar, little-endian:

```
off  type                     field
  0  char[8]                  "DS4IMGE1"
  8  int32                    n_embd          must equal w_.n_embd
 12  int32                    n_tokens        merged image tokens
 16  int32                    n_palette
 20  int32                    flags (0)
 24  int32[n_palette]         palette         token IDs, in cycle order
 ..  float32[n*n_embd]        data            row-major, n_embd fast
```

`n_embd`-fast matches `CpuEmbedder::embed`'s output layout, so the splice is a
memcpy. The palette ships **with** the embeddings on purpose: it is part of the
trained artifact, and a hardcoded engine copy could drift from the model served.

**Placement.** `src/server/main.c:2239` encodes the whole prompt in one call:

```c
n_prompt = ember_backend_encode(be, req->raw_prompt, &ids);
```

Two workable mechanisms, in preference order:

1. **Split-encode on a sentinel.** Split `raw_prompt` on a literal `<image>`,
   encode prefix, emit `n_tokens` palette IDs, encode suffix. Exact and
   tokenizer-independent. `main.c:683` already documents this pattern
   ("append a separately-tokenized suffix … BPE may merge across the boundary"),
   so it is the idiom this codebase already uses. Costs one setter to carry the
   span down to the backend.
2. **Marker-run detection.** The harness repeats a marker string `n_tokens` times
   and the backend finds the run, substituting in place — no span threading and
   no `main.c` change. Cheaper, but it depends on the marker tokenizing 1:1 under
   repetition, which BPE does not guarantee. **If this is used it must verify the
   run length equals `n_tokens` and abort loudly otherwise**; a short run would
   silently misalign every image token.

**Splice.** In the prefill loop, for chunk start `i` and each `t < n_tok`, with
absolute position `p = i + t` inside the image span, overwrite
`embed[t * n_embd ...]` with row `p - img_start` of the sidecar data.

Respect two invariants from upstream, both load-bearing:

- **Phase is anchored to the image start**, `(absolute_position - image_start) % 64`
  — not to the chunk. Ember chunks prefill, so getting this wrong makes routing
  depend on chunk boundaries, i.e. output that changes with batching.
- **Prefill only.** Decode tokens keep their ordinary IDs.

**Gate it behind an env var** (e.g. `EMBER_DS4_IMAGE_EMBED`) and document it as a
validation path: one image, one request at a time, not concurrency-safe.

## Still true, and still worth fixing

**Ember accepts image parts and silently ignores them.** Posting an `image_url`
returns a normal completion rather than an error, so a harness can believe vision
works. Mainline llama.cpp returns `HTTP 500: image input is not supported`.
Whatever lands here, make the unsupported path *loud* — a silent drop is worse
than a missing feature.

## Running the gate

Needs the full model resident, so it needs a maintenance window on the host —
production holds ~109 of 125 GB and there is no room for a second instance. Batch
it with any other queued experiment that needs the same window.

Run it twice: against the BF16 intermediate first (stage 2 — is the artifact
sound?), then against the 2.58 BPW quant (stage 3 — did quantization break the
alignment?). Comparing the two is the entire point; a single run against the
quant cannot distinguish "upstream is broken" from "our quantization broke it".

## Fallback

Mainline llama.cpp already runs MoonViT. A DS4 GGUF served by patched mainline
llama.cpp is a fallback if this port stalls — at the cost of the ROCMFPx types
and DSpark.
