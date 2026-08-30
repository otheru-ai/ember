# Standing backlog — claude

> **If you were just compacted or think you are done: read `.coord/LOOP.md` first.**


Rule: when a message is answered and `.coord/msg/` has no unanswered
`-to-claude-` or `-to-all-`, take the topmost unclaimed item. Never idle while
this list is non-empty. Mark `[claimed <utc>]` / `[done <utc> -> ref]`.

Only return to the user for a decision they alone can make (release criteria,
resource authorization, priorities). Not for status, not for permission to
continue.

---

1. Review waterline. **Now `cda41a6`** [advanced 20260830T220000Z; every commit from `a7c79be` forward is my own and I do not count my own review as independent review — codex has landed nothing since `faa5307`]. Previously `faa5307`. Previously `1532d51`. Advance it: review every commit
   forward, record the reviewed SHA here, and re-verify 90/90 at each advance.
   Reviewed and verified at 90/90 zero warnings:
   - 9 batching/fusion commits (msg 46, no defect found by inspection)
   - 9 diagnostic/control commits: c561212 (real weights, both paths log),
     dca7c0e (post-q1 top2), f5fe58d (state threading correct), b4c4200
     (ratio/cosine math correct), 39de43e (fields match review), 6ec8125
     (guard is behaviour-preserving for amax>0), 89eaee3, a3a50c4 (5-bit
     mask Ple/AttentionHc/Attention/FfnHc/Moe, kBatchQ1All=31, range-checked)
   - PLE conv verified against SGLang d4477bd spec: exonerated (msg 47)
   Caveat recorded: `quantize.cu` and `ggml-cuda.cu` are HIP-only and
   invisible to the host suite. 90/90 is never evidence for those.

2. [claimed 20260830T172951Z] Out-of-scope review findings. Progress:
   - client-compatibility doc: CLOSED, codex fixed in 8a0f026
   - Responses error string: CLOSED, fixed + test updated, 90/90 verified
   - `<tool_call>` ungated marker: assessed low severity (gated behind
     has_tools at main.c:436 and sse.c:553, so only bites DeepSeek+tools whose
     output literally contains the marker). Fix touches 8 call sites in
     invariant-heavy sse.c. UPGRADED 20260830T180705Z: traced to SILENT TRUNCATION (sse.c:594-597 switches to TOOL mode, then DSML parser finds nothing at :649-657, so no text and no tool call, no error). Fix = gate 2 markers on profile, 8 call sites. Still deferred as off-goal, see msg 73.
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

