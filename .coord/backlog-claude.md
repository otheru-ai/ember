# Standing backlog — claude

> **If you were just compacted or think you are done: read `.coord/LOOP.md` first.**


Rule: when a message is answered and `.coord/msg/` has no unanswered
`-to-claude-` or `-to-all-`, take the topmost unclaimed item. Never idle while
this list is non-empty. Mark `[claimed <utc>]` / `[done <utc> -> ref]`.

Only return to the user for a decision they alone can make (release criteria,
resource authorization, priorities). Not for status, not for permission to
continue.

---

1. Review waterline. **Now `a3a50c4`** [advanced 20260830T171312Z]. Previously `1532d51`. Advance it: review every commit
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
     invariant-heavy sse.c. DEFERRED, not dropped.
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
   [20260830T173943Z] copyBufferRect counted: ZERO (trace grep). No undercount. All 1.27M are 1D packed -> src1 is a 2D-packed slice with inconsistent nb2/nb3. sync_fallback refuted (0 of 4924 dispatches). Remaining: pair copies to producing op via Correlation_Id.

6. [done 20260830T173524Z -> msg 59, verified grok 83 against ggml-cuda.cu:1478-1510] [claimed 20260830T173428Z] Standing: when grok or kimi files a result, check it against source before
   relaying to codex. Grok's PLE spec was verifiable in one read and produced
   an elimination; that pattern is worth repeating rather than forwarding
   unchecked.
