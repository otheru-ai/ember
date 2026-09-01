# ds4-server Fidelity Audit — ember vs antirez/ds4

**Date:** 2026-07-25
**Reference (source of truth):** `antirez/ds4` (pure C), `/srv/lucebox/ds4-src` — `ds4_server.c`, `ds4_eval.c`, `ds4_kvstore.c`, `ds4.c`, `ds4_agent.c`, `rax.c`.
**Reimplementation:** ember C server, `/home/glovepost/ember/src`.
**Method:** six parallel subsystem audits, each reading the ds4 source exhaustively and comparing to the ember counterpart. All citations are `file:line`.

## Verdict legend
- **FIX** — a divergence from ds4 that is a bug or reduces fidelity; align to ds4.
- **KEEP** — ember diverges *intentionally* (a lucebox performance fix or leak-fix); do **not** revert.
- **DECIDE** — a behavioral default that changes production semantics; needs an explicit call.

---

## KEEP — intentional deviations (do NOT "align" these back to ds4)
These were flagged by the agents specifically so they are not mistakenly reverted:
- **Terminator hint** (verbose "Considering the limited time…\n</think>\n\n") instead of ds4's bare `</think>` — `model_card.c:15-17`. ds4's bare close made DeepSeek-V4-Flash emit empty content → reasoning leaked to Telegram. KEEP.
- **`hard_limit_reply_budget = 1024`** (ds4 uses 512 hard / 1024 soft) — `model_card.c:14`. Deliberately raised 2026-07-24 with the hint fix. KEEP.
- **Usage extensions** `timings`/`accept_rate`/`restored_prefix` — `main.c`, `sse.c`. KEEP.
- **Buffer-and-resplit streaming, UTF-8 stream-safe holdback, marker-safe holdback** — faithful and correct. KEEP.
- **Disk-backed cross-restart KV cache, token-id prefix match, expert_top_k=4, DSpark** — KEEP.
- **Single coarse `gen_lock`** (vs ds4 batched fine-grained locks) — correct for ember's single-slot model. The ds4 "batched session recovery race" fix (slot-binding + 409) is **N/A** to ember and must NOT be ported (it would be dead code). KEEP.

---

## CRITICAL — break real DeepSeek-V4 traffic

### C1. Non-streaming tool calls leak raw DSML into `content`; no `tool_calls`, wrong `finish_reason`
`main.c:184-256` only splits `<think>…</think>`; the remainder (incl. `<｜DSML｜tool_calls｜>…`) is escaped verbatim into `content`, and `finish_reason` is `res.finish_reason` (only "stop"/"length"). ds4: `ds4_server.c:11619-11623` sets `finish="tool_calls"`, `append_tool_calls_json` (`5270-5288`) emits a structured array after stripping DSML. The streaming path *does* handle tools (`sse.c:198-224`) — so the two paths disagree. **FIX:** run `ember_parse_dsml_tool_calls` in the non-stream branch, strip the block from `content`, emit `tool_calls` + `finish_reason:"tool_calls"`.

### C2. Assistant-turn thinking render is structurally broken (unclosed `<think>`, content mislabeled)
`chat_template.c:99-104` renders a history assistant turn as `<｜Assistant｜>` + (`enable_thinking?"<think>":"</think>"`) + `content` + EOS. In think mode this yields `<｜Assistant｜><think>{content}<｜end▁of▁sentence｜>` — **`<think>` never closed, visible content trapped inside the think block**. ds4 (`ds4_server.c:2478-2498`): open `<think>` + reasoning + `</think>` + content, with a collapse rule (`tool_context || i > last_user_idx` keeps reasoning; older turns drop it). Corrupts every multi-turn reasoning conversation. **FIX** (needs C4).

### C3. Tool preamble bytes differ + think-hint is conditional → breaks KV-prefix reuse
`chat_template.c:20-53,73-76`: reworded intro, one invoke stanza (vs ds4's two), placeholders `$NAME/$P/$VALUE` (vs `$TOOL_NAME/$PARAMETER_NAME/$PARAMETER_VALUE`), omits the entity-escaping paragraph, and emits the think hint **conditionally on `enable_thinking`**. ds4 (`ds4_server.c:2063-2088`) emits **both** think sentences unconditionally. Because ember's preamble changes with think mode, the prompt prefix differs between thinking/non-thinking requests → tokenization diverges from ds4 on every tool call and KV-prefix reuse breaks. **FIX:** port `append_tools_prompt_text` byte-for-byte; make the think-hint block constant.

### C4. `ember_chat_msg` lacks `reasoning` and `calls` fields
`chat_api.h:11-15` has only `role/content/name`. ds4 parses `reasoning_content`→`msg.reasoning` (`1684-1689`) and `tool_calls`→`msg.calls` (`1698-1703`) and replays them. Without these, assistant history can never carry reasoning or replay prior tool calls (multi-turn agent loops see only tool results, never the model's own prior calls). **FIX:** add fields, parse them, render assistant calls as DSML (enables C2 + tool-history).

---

## HIGH

### H1. Thinking default inverted; `reasoning_effort:"none"` ENABLES thinking  → also disables force-close  *(DECIDE on default-on)*
`main.c:100`: `enable_thinking = (req->reasoning_effort != NULL)`. So no-effort → thinking OFF (ds4 default is ON), and `"none"` (non-NULL) → thinking ON (ds4: `"none"`→`DS4_THINK_NONE`→OFF, `ds4_server.c:809-833`). The inversion also silently disables the whole force-close budget for default requests (armed only when `enable_thinking`, `main.c:122`). **FIX** the `"none"` inversion and unknown-value handling unconditionally. **DECIDE** whether absent-effort should default thinking ON (ds4) or OFF.

### H2. Reasoning-effort tiers are DEAD CODE — `low`/`high`/`max` have no runtime effect
`ember_model_card_think_budget` + `.tiers` (`model_card.c:18-19,88-95`) have zero callers; `main.c` never computes a phase-1 think cap. ds4 itself has no numeric tiers — effort selects a prompt mode (HIGH/MAX) + (MAX) a prefix, ctx-gated (`ds4_server.c:814-835`, `ds4.c:379-387`). **FIX:** thread a resolved NONE/HIGH/MAX mode into rendering (H7); either wire or delete the numeric tiers.

### H3. Sampler surface dropped — `top_p`/`top_k`/`min_p`/`seed` unparsed & unthreaded; card `top_k:40`/`min_p` dropped
`chat_api.c:32-73` parses none of them; `ember_gen_request` (`ember_backend.h:51-77`) has no `top_k/min_p/seed`; `main.c` never sets `greq.top_p` (backend falls back to 1.0, `backend_dflash.cc:132`); generation is always unseeded. `model_card.c:63-67` reads only temp/top_p, dropping the card's `top_k:40`/`min_p`. ds4 parses+threads all four (`ds4_server.c:3018-3046,11398`). **FIX:** parse into request, add ABI fields, thread through the bridge incl. seed.

### H4. `stop` sequences completely ignored
Not parsed (`chat_api.c`), not enforced anywhere. ds4 parses string|array (`ds4_server.c:981-1008`), holds back stream-safe (`stop_list_stream_safe_len`, `1032`), truncates + `finish="stop"` on hit (`11607-11617`). **FIX:** parse `stop`, implement stop-scan + truncation in stream and non-stream.

### H5. Marker detector/parser asymmetry → plain-XML tool calls detected-but-dropped
Stream detector `TOOL_STARTS` (`sse.c:27-33`) matches `<tool_call>`/`<tool_calls>` but `tool_parser.c:19-30` has no plain-XML parse family (it has an ascii-`?` family ds4 never emits). On plain-XML output ember suppresses content AND drops the call. ds4 parses full/short/plain-XML (`ds4_server.c:4548-4575`). **FIX:** replace the ascii-`?` family with ds4's plain-XML family; keep detector ⊆ parser.

### H6. Blocking socket writes, no stall timeout → a stuck client freezes the whole single-slot server
`http.c:60-69` is a plain blocking `send()`; client fd never set non-blocking; `run_chat` holds `gen_lock` across the whole stream (`main.c:98-262`). A client with a full TCP window makes `send()` block forever inside `on_token`, holding `gen_lock` and freezing every other request — the exact class of contention seen during cutover. ds4 sets the client non-blocking with a 2s send-stall deadline + 10s IO timeout (`ds4_server.c:4473-4498,12500-12514`). **FIX:** non-blocking client fd + `POLLOUT` stall deadline in `ember_send_all`; `SO_RCVTIMEO/SO_SNDTIMEO`.

### H7. THINK_MAX reasoning-effort prefix + "Reasoning Effort" directive missing
ds4 injects `ds4_think_max_prefix()` after BOS on MAX (`ds4_server.c:2457`, `ds4.c:379-382`, ctx-gated ≥393216) and a "Reasoning Effort: High/Max" system directive (`2543-2545`). ember has no MAX mode. **FIX** with H2.

### H8. Tool schema serialized by re-dump (number reformatting) instead of raw client bytes
`chat_template.c:34-53` re-parses and `ember_json_dump`s each `function` → integers `%lld`, floats `%g` (`json.c:252-255`), whitespace stripped; not byte-identical. ds4 passes the client's raw JSON substring (`ds4_server.c:1227-1255`). **FIX:** raw-passthrough the `function` object; join single `\n`, no trailing newline.

### H9. No `context_length_exceeded` pre-check / 400
`main.c:283-298` has no prompt-vs-ctx guard. ds4 returns a structured 400 `{...code:"context_length_exceeded",n_prompt_tokens,n_ctx}` (`ds4_server.c:5368-5400,12438`). **FIX.**

### H10. Default `max_tokens` 2048 vs ds4 full-context
`main.c:113`, `backend_dflash.cc:129` default 2048; a reasoning turn routinely exceeds it → premature `finish:"length"`. ds4 defaults to `393216` capped to ctx room (`ds4_server.c:12670,11356`). **FIX:** default to context room. *(Also clamp negative max_tokens→0 to match ds4.)*

### H11. `ember_kv_reserve` round-robin hands out live-owned slots → KV corruption on non-committing gens
`kv_cache.c:108-119`: when `n_entries<cap` it returns `next_slot++%cap` without checking the slot is free or consulting the eviction policy. After "holes" (reserved-but-not-committed slots from cancelled/disconnected gens, `main.c:167,193`), it can return a slot a live entry owns — and if that gen writes an inline snapshot then fails (no commit), a stale map entry points at overwritten KV → later restore of wrong KV. **FIX:** in the `<cap` branch pick a genuinely free slot id (or always allocate via `evict_victim`); defer capacity eviction to commit time.

---

## MEDIUM

- **Initial `delta.role:"assistant"` chunk missing** — `sse.c`/`main.c:151-179`; ds4 `11314-11319`. Some OpenAI SDKs require it. FIX.
- **Streaming usage chunk always emitted** — should gate on `stream_options.include_usage`; ember never parses `stream_options` (`sse.c:236-246`; ds4 `5489-5491`). FIX.
- **DSML tool-separator whitespace not trimmed** — leaks `\n\n` into content before a marker (`sse.c:47-74`; ds4 `trim_tool_separator_ws` `4607-4610`). FIX.
- **No SSE `event: error`** — on `!res.ok` ember still emits a normal finish (`main.c:166-179`; ds4 `5418-5433`). FIX.
- **No `SO_RCVTIMEO`/body cap** — slowloris/unbounded-alloc exposure (`http.c`; ds4 `12500-12506,12242`). FIX (with H6).
- **DSML entity unescaping missing** — `&lt;`/`&amp;` not decoded in attrs/`string="true"` values (`tool_parser.c:42-51,136`; ds4 `dsml_unescape_text` `4632-4671`). FIX.
- **`tool_choice` ignored** — `"none"` still renders tools (`chat_api.c:56-59`; ds4 `2984-2997`). FIX.
- **`tool_result` sentinel not escaped** — embedded `</tool_result>` breaks framing (`chat_template.c:93-98`; ds4 `append_tool_result_text` `2240-2256`). FIX.
- **`string="false"` handling inverted** — ember JSON-string-escapes non-scalar-looking values; ds4 always emits raw minified JSON, `null` on empty (`tool_parser.c:132-137`; ds4 `4673-4684`). FIX.
- **`developer` role dropped** — not treated as system (`chat_template.c:64,87`; ds4 `role_is_system` `2408-2410`). FIX.
- **Assistant header / gen-prompt gating** — ember always emits `<｜Assistant｜>` + opener; ds4 gates on `pending_assistant`, so a history ending in an assistant turn gets a spurious second opener (`chat_template.c:99-111`; ds4 `2460,2501-2504`). FIX.
- **Prefill keepalive not throttled** — emits per-callback; ds4 caps at 5s (`main.c:84-92`; ds4 `10439-10446`). FIX.
- **`snap_slot == restore_slot` unguarded** — `main.c:133-149`; can alias, snapshot into the restore source. FIX.
- **Query string not stripped from path** — `/v1/models?x=1` 404s (`http.c`/`main.c:269-309`; ds4 `12262-12263`). FIX.
- **400 body missing `type`; generic message** — `main.c:292-293`; ds4 `5345-5353`. FIX.

## LOW / informational
- `id` scheme `chatcmpl_%08lx(unix)` collides within a second (`main.c:107`); ds4 uses a monotonic counter. FIX (cheap).
- `usage` omits `prompt_tokens_details.cached_tokens` (ember has `restore_len`). Optional.
- Incremental tool-call arg deltas (ds4 streams header+arg fragments); ember emits whole call at finish. Optional.
- No early-stop at tool-calls end marker (ds4 halts decode; `ds4_server.c:11619`). Optional perf.
- No truncated-DSML repair (`try_repair_dsml`, ds4 `5130`). Optional robustness.
- Snapshot-point selection is ember-original (turn-boundary vs ds4 trim/align+periodic). Not buggy; add trim/align rounding for closer parity if desired.

---

## DECIDE (change production semantics — need an explicit call)
1. **Default temperature** for requests that omit it: card **0.6** (DeepSeek-recommended, current) vs ds4 **1.0** vs production greedy **0** (required for DSpark speculative fast path). Note: default 0.6 means default requests do *not* hit the greedy DSpark path.
2. **Thinking-on-by-default** (ds4) vs current **off-unless-`reasoning_effort`**. Hermes always sends `reasoning_effort`, so this only affects other/direct clients.

Independent of the above, the `"none"`-inversion and unknown-effort handling (H1) are unconditional bugs to fix regardless of the default choice.

---

## Status (applied 2026-07-25, commit 1b196b0)
Decisions taken: **default temperature 1.0, thinking ON** (both ds4).

**FIXED:** all CRITICAL (C1–C4); HIGH H1–H11; MED: initial role chunk, usage-chunk
gating on `stream_options.include_usage`, tool-separator whitespace trim, developer
role, tool_result sentinel escape, `tool_choice:"none"`, monotonic id, 400 `type`,
`Connection: close`, `snap_slot != restore_slot` guard, query-string strip, body cap,
socket timeouts. Verified: 221/221 QA, render harness (byte-exact template), real-backend
GPU run (default-think, 33-tool, streaming role chunk, sampler params, ctx-400).

**KEPT (intentional):** terminator hint, `reply_budget=1024`, timings/accept_rate,
buffer-and-resplit, disk KV, expert_top_k, DSpark. Session-recovery race is N/A (single-slot).

## Deferred items — worked through (commits 09271e2, 19e3632)
**NOW DONE:**
- **Sampling penalties** (`repetition_penalty`/`rep_window`, `frequency_penalty`,
  `presence_penalty`) parsed + threaded via the bridge to `SamplerCfg` (verified:
  penalties → sampled path, greedy → DSpark).
- **SSE `event: error`** on backend `!ok` + non-stream 500, using the surfaced
  `error_code`/`error_detail`; `degenerate_decode_close` → `finish_reason:"length"`;
  `budget_forced_close`/`empty_visible_output`/`spec_decode_ran` in `usage.backend`.
- **Stream-safe `stop`** holdback in the SSE path (hold `max_stop_len-1`, cut at the
  earliest complete stop) + `on_token` halt; `finish_reason:"stop"`.
- **Tool-calls end-marker early-stop** (`on_token` halts at `</...tool_calls>`).
- **Prefill keepalive throttled to ~4s.**
- **`prompt_tokens_details.cached_tokens`** in stream + non-stream usage.
- **Truncated-DSML repair** (conservative: only with a complete parameter present).
- **Schema number fidelity** — `ember_json` preserves the raw number token, so dump is
  byte-exact; tool schemas now reach the model verbatim (no `%g`/`%lld` reformatting).

- **Incremental tool-call argument deltas** — tool calls now stream as a header delta
  (id/name) + `function.arguments` fragments per parameter (ds4 start-delta + args_delta),
  replacing emit-at-finish. Verified byte-identical to the batch parse; real-model run
  streamed 1 header + 4 arg fragments → valid JSON, `finish=tool_calls`.

- **`min_p` enforcement** — added a `min_p` field to lucebox's `SamplerCfg` and the
  CPU sampler chain (`prob >= min_p·max_prob` floor after softmax; GPU sampler falls
  back to CPU when set), rebuilt `dflash_common` + relinked `ember-dflash`. Verified:
  min_p=0.90 @ temp=2.0 stays coherent ("Paris") while min_p=0.00 @ temp=2.0 is
  gibberish. lucebox's existing binary is untouched (static lib; only `dflash_common`
  rebuilt). This was the last remaining item.

**Nothing deferred — every audit item is implemented.**
