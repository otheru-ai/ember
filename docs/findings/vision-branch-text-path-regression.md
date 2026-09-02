# The vision work put a non-inlinable call in the prefill mask loop

Found 2026-09-02 while tracing whether the vision changes could explain row C.
**This is a text-path regression introduced by vision support**, present on
requests that contain no images at all.

## The change

`build_mla_attention`'s host-side causal mask construction, on the
`layer_major_batch` branch — the prefill path:

```diff
     for (int r = 0; r < n_prior_rows; ++r) {
         const int prior_pos = kv_start - n_prior_rows + r;
-        if (prior_pos < min_pos) col[r] = -1e30f;
+        if (!dflash::deepseek4_raw_attention_visible(
+                pos_i, prior_pos, w.n_swa,
+                visible_left, visible_right)) {
+            col[r] = -1e30f;
+        }
     }
```

and the same substitution in the loop immediately below it.

One inline integer comparison became a call to a function **defined in another
translation unit** (`deepseek4_vision_contract.cpp:483`, declared without a body
in `deepseek4_vision_contract.h:158`). The build has **no LTO/IPO**, so it
cannot be inlined, and a call in the loop body also prevents vectorisation of
the whole loop.

The function itself is heavier than what it replaced: five validity comparisons,
two int64 subtractions, `std::min`, `std::max`, and two int64 comparisons,
against one `int` compare before.

## Semantics are unchanged for text

With `visible_left = visible_right = 0`, which is what a text request passes,
`image_start` collapses to `query_pos`, `start` becomes `ordinary_start`, and the
predicate reduces to the old sliding-window rule plus a causal upper bound. So
this is **pure cost, not a behaviour change**, on the text path.

## Measured

Microbenchmark at representative sizes (2048 tokens x 4096 mask columns,
`-O2`, same machine, five runs):

| variant | per layer-chunk | x43 layers |
|---|---:|---:|
| old inline compare | 3.40 ms | 146 ms |
| **current cross-TU call** | **11.15 ms** | **479 ms** |
| same function, made `inline` | 2.52 ms | 108 ms |

**6x slower per element**, and the fix recovers all of it — slightly better than
the original, because the inlined form vectorises.

Caveat on magnitude: 4096 mask columns is representative, not measured from a
production chunk. `n_attn = n_raw + n_comp_attn + n_old_rows`, so the real
per-chunk cost depends on the actual window and chunk size. The mechanism and
the ratio are the findings; the absolute milliseconds are illustrative.

## Why it fits row C

Row C regressed **prefill at every depth >= 1024** (-1.0% to -3.2%) while
autoregressive decode was uniformly *faster* (+0.4%). This mask loop runs during
prefill and not during single-token decode, and its cost scales with context —
which is the shape observed.

It does not explain everything. Row C also showed decode -9.00% at depth 1024,
which this cannot account for.

## Fix

Move `deepseek4_raw_attention_visible` into the header as `inline`. No semantic
change, no ABI concern (it is a leaf predicate), and the measurement above shows
full recovery. Leave the out-of-line definition for any other caller, or drop it.

## Second, smaller item found in the same trace

`deepseek4_graph.cpp:5697` copies the entire token array on **every** graph
build before checking whether any vision token exists:

```cpp
vision_ids.assign(token_ids, token_ids + n_tokens);
for (int32_t token : vision_ids) { if (token >= w.n_vocab) { ... break; } }
```

`ds4_has_vision_tokens()` at line 73 performs the same detection with
`std::any_of` and no allocation, and is used at two other sites. This one is a
heap allocation plus a copy per graph build on the text path. Small beside the
mask loop, but free to remove.
