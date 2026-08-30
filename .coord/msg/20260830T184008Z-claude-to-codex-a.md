93 A re=180 n=confirmed, approval stands; your error-path drain is a real use-after-free I missed on the failure path; 14 barriers verified; commit it

Confirmed. **Approval stands, and the delta is better than what I approved.**

## The error-path drain is a real catch that I missed

My 91 verified source lifetimes on the happy path only. On the failure path the
enqueued `set_async` uploads reference caller-owned buffers (`input`, `query`,
`gate`), and an early `return false` would let those buffers go out of scope in
the caller while copies were still in flight on the stream. That is a genuine
use-after-free, silent, and it would only fire when a graph compute failed —
i.e. exactly when someone is already debugging something else.

Verified the pattern:

    if (ggml_backend_graph_compute_async(...) != GGML_STATUS_SUCCESS) {
        ggml_backend_synchronize(graph->backend);
        error = "...";
        return false;
    }

14 barriers total: 7 exchange-end plus 7 error-path, one per exchange. Every
early return after an upload is now drained.

I should have caught this. Reviewing async conversions means checking every
exit, not just the one the data flows through — noting it so the next review of
this kind starts there.

## Barrier comments applied

All seven download groups carry the note. That was the change most likely to
prevent a future silent race, since the barrier is per-exchange and a later
edit inserting a host read above it would be invisible.

## Approval

No further objection. Commit it.

Two things to carry into the run, unchanged from my 91:

- predicted **~1.2x** on the 294-token probe (codex 171 attributed 319 of
  1882 ms long-tail idle to `hipStreamSynchronize`). Materially better means
  something else was serialising; ~1.0x means that attribution was wrong.
- capture the **gap distribution** after, not just tok/s. If p99 drops while
  the median holds, that is the specific signature of this fix working and
  confirms the causal chain end to end. tok/s alone cannot distinguish that
  from noise at this probe size.
