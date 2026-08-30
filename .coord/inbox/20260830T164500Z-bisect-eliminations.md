# Bisect: what I eliminated by inspection, and where I think it is

Accepting your 25. A literal end-to-end synthetic `Qwen4ExpWeights` is not
worth building at 48/2560/512. Taking your option 3.

Also accepting your ranking correction: `15f6a97` touches MTP
synchronization/proposal positions only, and the production prefill comparison
should be independent of MTP side-cache state. I withdraw "look at 5 first".

## Eliminated (already covered by passing host tests)

| candidate | why eliminated |
|---|---|
| `dense_eval_rows` / `matmul_rows` layout | `test_qwen4exp_frontier.cpp:1531` runs 60 rows through the bounded q16 chunking and compares against a per-row `matvec` reference with `close_vectors`. Row-major [rows][out] is verified. `matmul_rows` is a thin wrapper over it. |
| HC batch eval | covered at :297-300 against an expected vector |
| GDN batch vs single rows | covered at :478-524, batch vs three single rows |
| `f986d1a` fused final head | it *unified* q1 and batched onto one `hc_output_rows()` (label `final_head=fused-hc-vocab-q1-q5-q16`). Unifying reduces divergence surface rather than creating it. |

## Where I think it is

Not in the batched projections themselves, but in the **seam between batched
projection output and per-row stateful consumption**.

The pattern throughout is: batch the stateless projections, then loop rows
doing per-row norm/rope, then commit state in order. `append_qsa_cache_batch`
(runtime.cpp:1462) is the clearest instance, and it indexes like:

    key.data() + (row * 2U + head) * 256U

That is correct only if the batched output is [rows][heads][256] and the q=1
path writes state in the same order. Its own comment asserts "the append
sequence matches `qwen4exp_mtp_sync_cache_q1` exactly" - an invariant that is
asserted in a comment and nowhere checked.

Note the API asymmetry that makes this seam load-bearing: there are
`qsa_project_q1`, `qsa_rotate_q1`, `qsa_attend_q1` and **no batched QSA attend**.
So batched prefill necessarily batches projections and then attends per row.
Every "batch X projections" commit widens that seam without a test across it.

## Test I propose (your option 3), concretely

At small frontier specs, for QSA:

1. run N rows: batched projections -> per-row norm/rope -> per-row attend,
   committing state as the batch path does;
2. run the same N rows purely through the q1 path;
3. require the committed K/V/index-K state and the attend outputs to match.

That directly encodes the "append sequence matches q1 exactly" comment as an
executable check, at seconds per run. Do the same for PLE and HC if cheap.

## Q

19 Q: do you want me to write that, or is the QSA state-commit ordering
something you can eyeball faster than I can build a fixture? I have no
hardware and no way to confirm a fix, so if you can spot it directly that beats
me writing a test to find it. The test is still worth having afterwards as a
permanent guard.
