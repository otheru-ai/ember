# Standing backlog — claude

> **If you were just compacted or think you are done: read `.coord/LOOP.md` first.**


Rule: when a message is answered and `.coord/msg/` has no unanswered
`-to-claude-` or `-to-all-`, take the topmost unclaimed item. Never idle while
this list is non-empty. Mark `[claimed <utc>]` / `[done <utc> -> ref]`.

Only return to the user for a decision they alone can make (release criteria,
resource authorization, priorities). Not for status, not for permission to
continue.

---

1. Review waterline. **Now `1ee72b8`** [advanced 20260831T070000Z]. Tranche 1
   (`1ee72b8`, resident QSA preparation) accepted after a FIRST-HAND build:
   `ember-rocm:10.0-dev` container per `AGENTS.md:100`, Release + `EMBER_ENGINE`
   + `EMBER_STRICT`, `build-claude-review/` — 0 warnings, frontier 126/0,
   oracle 23/0. My `set_rows` liveness finding was real: codex's OUTPUT flags
   exposed a second latent bug in the `current_key`/`current_value` readbacks.
   Prior waterline was `5ac6d95` [advanced 20260831T052000Z]. Codex's
   engine commits all independently verified rather than accepted on report:
   `5258cc6`, `5e7a31d`, `9f1dc33`, `86a5ce1`, `01b8218` (prefill margin
   criterion — verdict logic matches what I reviewed, exact-stream metrics are
   reporting only) and `5ac6d95` (Q4_K allow-list plus the opt-in HIP MMID
   dispatch test — all three of my review points addressed, including the
   telemetry latch ordering that would otherwise have voided the result).
   Strict ROCm build 4/4 no warnings, invariants pass, host 90/90, run by me.

2. [claimed 20260830T172951Z] Out-of-scope review findings. Progress:
   - client-compatibility doc: CLOSED, codex fixed in 8a0f026
   - Responses error string: CLOSED, fixed + test updated, 90/90 verified
   - `<tool_call>` ungated marker: assessed low severity (gated behind
     has_tools at main.c:436 and sse.c:553, so only bites DeepSeek+tools whose
     output literally contains the marker). Fix touches 8 call sites in
     invariant-heavy sse.c. UPGRADED 20260830T180705Z: traced to SILENT
     TRUNCATION. LOCATED PRECISELY 20260830T233000Z, and the earlier fix
     sketch was wrong:

     Mechanism: `TOOL_STARTS` (`sse.c:83`) carries `"<tool_call>"` and
     `TOOL_ENDS` (`:106`) `"</tool_call>"` **unconditionally**, so
     `ember_find_tool_start` fires on any text containing the Qwen marker.
     `:594-597` then switches the stream to `EMBER_SSE_TOOL` and moves
     `emit_pos` to `tool_start`. In `ember_sse_emit_tools` (`:649-657`),
     `st->qwen_tool_syntax` is false for a DeepSeek profile, so it calls
     `ember_parse_dsml_tool_calls`, which does not understand Qwen syntax and
     returns 0. `any_tool` stays false, nothing is emitted, no error is raised
     and the buffered text is gone.

     `qwen_tool_syntax` is set only by `ember_sse_set_qwen_tools` (`:266-269`),
     so the two markers are live on every profile while only one profile can
     parse them.

     **Do not** gate the markers as previously sketched: `ember_find_tool_start`
     / `_end` have **14 call sites** across `sse.c` and `main.c`, and several in
     `main.c` have no stream state to gate on. Threading a flag through them is
     a large change to buffer-and-resplit SSE, which `AGENTS.md` names as an
     invariant.

     Narrow fix instead, one site: in `ember_sse_emit_tools`, when the parser
     returns `n == 0`, the marker was a false positive — emit the buffered
     region from `st->tool_start` as ordinary content rather than dropping it.
     That converts silent truncation into correct passthrough without touching
     marker detection. **Before writing it**, read what `main.c` does with the
     `false` return, since that is the half I have not verified.

     **[done 20260831T071500Z -> `4926b93`]** Verified that half, and it
     CONFIRMS silent truncation while explaining why it is silent:
     `parse_executable_tool_calls` (`main.c:1282`) switches on
     `prompt_profile`, so on a DSML profile a Qwen marker gives
     `report.found == false` and it returns `true` at `:1316` — "ordinary
     assistant text". `stream_tools_valid` true makes the `invalid_tool_call`
     branch, the `dump_rejected_block` call and `ember_sse_discard_tool_block`
     ALL unreachable, so the turn ends "stop" with no error and no log. The
     defect is the asymmetry: `main.c`'s validator is profile-aware, `sse.c`'s
     marker detection is not.

     The narrow fix landed as sketched, and the `discard_tool_block` security
     rationale does NOT collide with it — that comment covers a block which
     PARSED and failed validation; here nothing parsed in any syntax family and
     the caller already classified the bytes as text. Delivering markup as
     visible content is additionally an outcome the server already accepts and
     counts (`note_tool_markup_leak`, `main.c:1698-1718`).

     Found a SECOND asymmetry in the same area while testing:
     `slice_has_tool_markup` (streaming, `:129`) recognises DSML only, while
     `ember_text_has_tool_markup` (buffered, `:142-145`) also checks
     `"<tool_call>"`. Deliberately did NOT extend the shared helper — that would
     make the `EMBER_SSE_TEXT` branch count the marker as a leak on requests
     carrying no tools at all, which is the false-firing `main.c:1704-1707`
     documents. Flagged at the new emit site only, where markup is present by
     construction.

     Also established: the buffered path never had this defect
     (`main.c:3155`, `tc.len == 0` + valid skips the trim), which gives the
     invariant the new test pins — streaming and buffered must agree.
     Host 90/90, `test_sse` 200/200.

     Still deferred while the correctness blocker is open: this is off the
     performance goal and sits in the most invariant-sensitive file in the
     server.
   Remaining:
   `flatten_content` unknown-block rejection + stale
   `docs/client-compatibility.md`; ungated `<tool_call>` marker in `sse.c`;
   Responses adapter error string; env-var-only vision provider config;
   W4A4 scale factor. All still open from the original branch review.

3. Verify every codex commit that lands, independently: build + ctest + read
   the diff. Do not accept a reported 90/90. Note that HIP-only sources
   (`quantize.cu`, `ggml-cuda.cu`) are invisible to the host suite - say so
   whenever a fix touches them.

4. [done 20260830T173428Z -> docs/qwen3.8-performance-status.md] Track the performance target. Ledger now durable in-repo with target, superseded measurement, bottleneck evidence, and the counter-unit correction. Update on each new measurement. Current: no trustworthy Qwen number exists;
   the only complete measurement (decode 4.498 / prefill 24.756 at `c5cb7a2`)
   predates the correctness validator. Target is DeepSeek-V4-Flash parity:
   prefill peak ~345 tok/s, decode 23.6-23.8 tok/s AR. Update when a valid
   measurement lands.

5. [claimed 20260830T172951Z] Copy classification. Progress:
   - mechanism identified: non-contiguous src1 materialize at
     `ggml-cuda.cu:1965`, 1.03:1 pairing with quantize (msg 44)
   - LARGER FINDING: `ggml_cuda_mul_mat_id` sync_fallback (`:2710-2762`) does
     two `cudaStreamSynchronize` + a host expert loop per MoE dispatch. Up to
     96 hard syncs/token. Confirmable with `DFLASH_MMID_TELEMETRY=1` (msg 52).
   [20260830T173943Z] copyBufferRect counted: ZERO (trace grep). No undercount. All 1.27M are 1D packed -> src1 is a 2D-packed slice with inconsistent nb2/nb3. sync_fallback refuted (0 of 4924 dispatches). [WITHDRAWN 20260830T191600Z] msg 62's "91.9% of copies precede quantize_q8_1" is **wrong and must not be reused**: it paired each copy with the next kernel by timestamp without filtering `Stream_Id`, so it measured co-occurrence, not adjacency. Codex's same-stream count is 37-39. Any argument resting on 91.9% is void. The surviving, independently measured facts are in `docs/qwen3.8-performance-status.md`: 15.6% GPU busy over a 2074-token prefill, and 95.55% of long-tail idle is late host submission.

6. [done 20260830T173524Z -> msg 59, verified grok 83 against ggml-cuda.cu:1478-1510] [claimed 20260830T173428Z] Standing: when grok or kimi files a result, check it against source before
   relaying to codex. Grok's PLE spec was verifiable in one read and produced
   an elimination; that pattern is worth repeating rather than forwarding
   unchecked.

7. [done 20260830T191500Z -> a7c79be, msg 213] CPU oracle for the graph RoPE
   path. `test/test_qwen_rope_graph_oracle.cpp`, ctest `qwen_rope_graph_oracle`,
   13/13 in `ember:qwen-faa5307-dev`, zero warnings. Result: **both** candidate
   mappings match `ember_qwen_yarn_apply` to ~1e-7, so Path 1 vs Path 2 is a
   cost question, not a correctness one. Mutation-tested, not assumed: reverting
   `c` to `inv_freq[k]` (my msg 99 error) gives max abs delta 0.852; `n_dims=256`
   gives 0.769 and is caught on the ext-factor path where ggml has no assert.
   Standing use: run it before and after any tranche 1 edit.

8. [done 20260830T200500Z -> 4e972da, msg 231] RMS half of the tranche 1
   oracle. `ggml_rms_norm` on the strided query-half view of
   `projected_query_gate`, matching the host reference to 1.19e-7. Asserts
   `ggml_is_contiguous_rows` on the actual view, which is what HIP's
   `supports_op` requires (`ggml-cuda.cu:5487-5492`). Mutation `nb[1]` for
   `nb[2]` gives 2.87 **and still passes contiguous_rows** — HIP would run it
   silently. Both halves of tranche 1 are now oracled.

9. Standing: the dead-code register `docs/dead-code-candidates.md` is mine to
   keep. User rule, recorded in LOOP.md. Entries need evidence with file:line,
   a falsifier, a scope (architecture / checkpoint / configuration), and a
   recommendation. **Before quoting any count in a performance argument, check
   it against the register** — the rotation stage cost us a 14-vs-12 barrier
   error already.

10. Open: I have now twice mis-stated tranche 1's payoff (msg 219, msg 227),
   both times by naming a stage instead of following what reads the buffer.
   Before asserting that a change removes a copy or a barrier, grep for every
   consumer of the buffer and cite the lines. Grok caught both.

11. Review finding, deferred until the correctness blocker closes (msg 247):
   `qwen4exp_runtime.cpp:1909-1917` pushes every batch row's position into
   `state.mrope_positions` before the layer loop, while the serial path pushes
   one per token at `:1161-1162`. Within a batch the position history runs
   ahead of the KV state, which the comment at `:1918-1921` implicitly denies.
   Inert at ctx <= 2048 — only the `!dense_selection` scorer reads it — but it
   weakens the bounds guard at `:832-836` from a real check into one that
   always passes. Fix: move the push to the per-row point where `index_key` is
   appended. Do not land while the blocker is open.

12. Deferred hardening, after the blocker closes (msg 285): `gated_delta_net.cu`
   computes one `gb_offset` and indexes **both** `g` and `beta` with **beta's**
   strides (`:519-527`, and the comment states it). The asserts check each is
   contiguous but not that they match. It is currently sound only because
   `qwen4exp_frontier.cpp:1005-1008` reshapes both to `[1, n_heads, n_tokens,
   1]`. Reshape them differently and it breaks silently, and only above n=1
   because the `t * sb2` term vanishes at q1. Add
   `GGML_ASSERT(ggml_are_same_stride(src_g, src_beta))` beside the existing
   asserts. Vendored file — record in `engine/VENDOR.md` if it lands.

13. [done 20260831T020000Z -> msg 303] Tranche 2 spec derived from the
   reference rather than from first principles.
   `docs/reference/qwen4exp_upstream.cpp:1029-1073` implements the device-side
   conv-state advance in fourteen lines, and our `ggml_concat` at
   `frontier.cpp:968` is already identical to theirs. Notable: upstream needs
   **no `retained_history` branch** — concat then take the last `state_cols`
   columns handles both `n < history` and `n >= history` by construction, where
   our host version at `:1164-1176` needs the two-branch splice. One decision is
   ours and not theirs: their `dst` is a different cache slot so there is no
   WAR, while our `conv_history` is a single resident tensor — use grok 211's
   two-buffer swap and mark it `set_input` + `set_output` per grok 199.

14. Standing, now that the reference exists: **derive each remaining tranche
   from `docs/reference/qwen4exp_upstream.cpp` before specifying it.** Tranche
   1 maps to their q/k/v path, tranche 3 to their device-side cache write.
   Reading a working implementation on the same silicon beats inferring one,
   and it is what turned tranche 2 from a design question into a mapping
   exercise. Do not copy wholesale — our snapshot/rollback contract has no
   counterpart there.

