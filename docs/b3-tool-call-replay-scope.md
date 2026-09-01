# B3 — Tool-call replay + KV continuation across tool turns (scoping)

Status: **scoping / design** (not implemented). Deferred from the tool-use review
as feature-sized. Reference implementation: DwarfStar (antirez ds4),
`/home/mythos/Projects/ds4/ds4_server.c` + `ds4_kvstore.c`.

## The problem

In an agentic loop Hermes calls a tool, gets a result, and continues — turn after
turn, on a growing history. Ember re-prefills that whole history **every tool
turn**, even though it already has working KV reuse for plain text conversations
(the multi-turn probe hit ~98% `cached_tokens`). The reason is narrow:

- Ember's history model (`ember_chat_msg`, `chat_api.h`) stores each tool call as
  canonical `name + arguments` and **discards the OpenAI `id` and the exact bytes
  the model sampled** (`raw` DSML). Tool-result messages' `tool_call_id` is also
  dropped.
- On the next turn `chat_template.c :append_dsml_tool_calls` **re-renders** the
  call from `name+arguments`. That canonical DSML is *not* byte-identical to what
  was sampled (whitespace, arg order, escaping, hidden `<think>`), so the
  **tokens diverge at the tool boundary**.
- Ember's prefix cache (`ember_kv_lookup`, `main.c`) is a longest-**token**-prefix
  match. A divergence at the tool call caps `restore_len` there → everything
  after (the tool result + the model's continuation) is re-prefilled.

The exact sampled bytes even exist at generation time in `g.acc` (`main.c
on_token`, delimited by `ember_find_tool_start/end`) — they're just thrown away.

Secondary gaps (same root cause): parallel calls can't be reliably associated
with their results (no ids), hidden reasoning that standard clients omit is lost,
and streamed ids repeat (already fixed in B5).

## Key insight — ember needs ds4's "Layer 1", not the whole subsystem

ds4 keeps **two** independent layers (confirmed by tracing):

1. **`tool_memory`** — a global `id → exact-sampled-DSML` map (rax `by_id`/`by_block`,
   refcounted block dedup, LRU bound by count+bytes, persisted to disk). A
   **pre-render substitution pass** (`tool_memory_attach_to_messages`, 8701)
   rewrites incoming canonical-JSON tool calls back to the exact sampled bytes
   *before* tokenizing. Per the reference: *this is what makes the re-rendered
   history byte-identical to what was sampled, so the prefix/disk matchers hit.*
2. **`live_tool_state`** — a per-slot in-RAM "live frontier" (`{valid,
   live_tokens, call_ids, visible_text}`) captured at `ds4_session_pos()` the
   instant the tool turn finished, leaving the KV physically parked there. On a
   matching next turn it appends **only** a re-tokenized `EOS + tool_result +
   next-assistant-prefix` suffix (`build_prompt_from_exact_prefix_and_text_suffix`,
   `ds4_kvstore.c:685`) and prefills just the diff.

**Ember already owns the equivalent of Layer 2** (prefix cache + backend
snapshot/restore + the disk prefix cache). What it lacks is **Layer 1**. Add the
id→exact-bytes map + substitution and ember's existing cache does the
continuation — no separate KV-checkpoint subsystem required.

So the work splits into two tiers:

- **Tier 1 (core, high value): exact-DSML replay.** Delivers the payoff through
  the cache ember already has.
- **Tier 2 (optional optimization): a live-frontier hot path** that skips even
  the token-prefix scan for the common "tool-result immediately follows the turn
  we just sampled" case, plus a 409 for naked tool-result requests with no
  binding. Nice-to-have; ember's single-worker/single-slot model makes the
  prefix cache already cover most of it.

## Design — Tier 1 (exact-DSML replay)

### 1. Data model (`chat_api.h/.c`, `tool_parser.h`)
- Add `char *id` to the per-call struct in `ember_tool_calls` (parsed from
  request `tool_calls[].id`, minted on generation).
- Add `char *tool_call_id` to `ember_chat_msg` (parsed from a `role:"tool"`
  message) so results associate with calls (parallel calls).
- Keep `raw` (the exact sampled DSML block) attachable to an assistant message's
  calls — the field the substitution pass fills, mirroring ds4 `raw_tool_text`.

### 2. Global id → exact-sampled-text map (new, e.g. `src/model/tool_memory.c`)
- Key: tool-call id (string). Value: exact sampled bytes for that assistant
  turn's tool block(s). Refcount/dedup so N parallel-call ids share one copy.
- Bound by entry count **and** total bytes, LRU-evicted. (ds4:
  `DS4_TOOL_MEMORY_DEFAULT_MAX_IDS=100000`, `MAX_BYTES=512MiB` — ember can start
  far smaller.)
- Lookup requires **all** ids in one message resolve to the same block (ds4 8734).

### 3. Capture at generation (`main.c`)
- When a tool call is finalized (the `ember_find_tool_start/end` block in
  `on_token`), keep the exact block bytes (and, for reasoning preservation, the
  visible assistant text incl. `<think>` up to that point).
- Mint 128-bit random ids (B5 already added `gen_call_id` via `/dev/urandom`;
  reuse it, check uniqueness against the map like ds4 `assign_tool_call_ids`).
- `tool_memory_put(id → exact bytes)` for every id in the turn.
- Return the id(s) in the response (`chat.completion` and the SSE tool-call
  header — the id must be the same one stored).

### 4. Pre-render substitution (`chat_template.c`)
- Before rendering history, for each assistant message whose `tool_calls` carry
  ids present in the map, attach the mapped exact bytes and render **those
  verbatim** instead of `append_dsml_tool_calls` canonicalization.
- **Trust/verify:** the map is server-authored, but the client controls which id
  it echoes. Verify the mapped block's function name(s) match the client's echoed
  `name` before substituting; on mismatch or eviction, fall back to canonical
  render (correct, just a cache miss). This blocks a client from binding a
  mismatched mapping.
- Reasoning preservation: if the mapped bytes include the sampled `<think>`, they
  restore hidden reasoning the client dropped (keeping tokens identical); if the
  client *did* send `reasoning_content`, prefer consistency (map wins when present).

### 5. Continuation falls out of the existing cache
Once the assistant tool-call tokens are byte-identical, `ember_kv_lookup` /
the disk prefix cache match through the tool boundary; `snap_cut` for the new turn
snapshots the extended prefix (call + result). No new KV code.

### Disk persistence (optional within Tier 1)
Ember has a disk prefix cache (`--kv-cache-dir`). Mirror ds4: append the
`id→DSML` entries whose bytes appear in a checkpoint to that checkpoint file, and
reload on lookup — so replay survives restart. Can be a later step; the in-RAM
map already delivers within-process value.

## Design — Tier 2 (optional live-frontier fast path)
- Per-request record `{valid, live_tokens (== snapshot cur_pos), call_ids set,
  visible_text}` captured when a tool turn finishes.
- If the next request is a tool-result continuation whose id-set == the stored
  set **and** `live_tokens == current frontier`, build
  `effective_prompt = [exact live prefix tokens] + [re-tokenized suffix]` (EOS +
  tool_result + next-assistant-prefix) and prefill only the suffix. **Re-tokenize
  the suffix — do not slice full-prompt tokens** (BPE can merge across the
  boundary; ds4 9488-9491).
- 409 fallback for a naked tool-result request whose binding is gone (ds4 11021).

## Phased plan
- **P1** — data model: parse+store `tool_calls[].id` and `tool_call_id`; thread
  the id through the response + SSE. *(low risk; no behavior change)*
- **P2** — `tool_memory` map (in-RAM, LRU, dedup); capture exact bytes at
  generation; mint+store+return ids.
- **P3** — substitution rendering in `chat_template` + name/arg verification +
  canonical fallback. **This is the payoff** — measure `cached_tokens` across a
  tool loop; expect the jump the text case already shows.
- **P4** — reasoning preservation, parallel-call set semantics, disk persistence.
- **P5 (optional)** — Tier 2 live-frontier + 409.

## Effort & risk
- Tier 1 (P1–P4): **medium**. New ~200–400 LOC `tool_memory` + touches
  `chat_api`, `chat_template`, `main.c`, `sse.c`. Mostly self-contained, server-
  side (reaches prod at standalone cutover). Stub-testable end to end.
- Tier 2: **medium-small** but more subtle (frontier bookkeeping, BPE boundary).
- Risks: token-identity fragility → **always degrade to canonical render/cold
  prefill** (never wrong, only slower). Client-controlled ids → **verify
  name/args before substituting**. Map eviction → graceful fallback. Disk-map
  correctness → gate behind the existing disk-cache identity fingerprint (#1).

## Validation
A tool-loop harness: user → assistant tool_call → tool result → assistant … for
~4 turns on a large system prompt, echoing back the server-returned ids.
`cached_tokens` on turns ≥2 should rise from near-0 (today) toward the ~98% the
text multi-turn case reaches. Also assert byte-identity: the substituted render
tokenizes to the same ids that were sampled.

## Recommendation
Implement **Tier 1** (P1–P4). It closes the actual gap that costs Hermes a full
re-prefill every tool turn, reuses ember's existing cache, and is bounded/stub-
testable. Treat **Tier 2** as a follow-on only if profiling shows the token-prefix
scan or the naked-tool-result case still matters after Tier 1.
