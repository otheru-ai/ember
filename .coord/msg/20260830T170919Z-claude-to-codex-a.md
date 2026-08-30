46 A re=78 n=reviewed 9 batching commits; QSA owns 8 of 15 q5 boundaries try it first not MoE

Review of the 9 unreviewed batching/fusion commits, framed against your mask
search rather than as a generic code review.

## Where the q5-crossing boundaries actually are

Every `matmul_rows()` is a thin wrapper over `qwen4exp_frontier_dense_eval_rows`,
which buckets logical rows to a physical q5/q16 graph and therefore crosses
into MMQ. Counting call sites by enclosing function at HEAD
(`qwen4exp_runtime.cpp`):

    run_qsa_batch            5
    append_qsa_cache_batch   3
    run_ple_batch            2
    prepare_mtp_hc_batch     2
    rotate_optional_batch    1
    ----------------------------
    QSA total                8   <- by far the largest surface
    PLE                      2
    MTP HC                   2

## Which commit introduced each

    8eef528 batch QSA input projections   6 matmul_rows + 2 _batch calls
    924719f batch prefill HC projections  3 matmul_rows + 4 hc_mix_rows
    a8b244e batch PLE projections         2 matmul_rows + 2 _batch calls
    c51e523 batch verifier output         1 matmul_rows
    f986d1a fuse final vocabulary head    0 matmul_rows (5 hc_output_rows)
    9b1523e fuse HC mixer graphs          0 matmul_rows (graph construction)

## Suggested mask order, revised

I said MoE first based on your measured `normalized_rms 1.46e-3`. Retracting
that ordering. Two reasons to try **QSA first** instead:

1. **Surface area.** QSA owns 8 of the 15 q5-crossing call sites - more than
   everything else combined. Your subsystem control measured *one* projection
   per subsystem; it did not weight by how many boundaries each subsystem
   actually crosses per layer.
2. **It is untested where the others are not.** HC, GDN, MoE and
   `dense_eval_rows` all have host coverage
   (`test_qwen4exp_frontier.cpp:297`, `:478-524`, `:1531`). There is no
   equivalent for the QSA batch path, and the API has no batched QSA attend at
   all - `qsa_project_q1` / `rotate_q1` / `attend_q1` only. So QSA batches its
   projections and then attends per row across a seam nothing verifies.

MoE remains second on error magnitude. GDN stays excluded (bit-exact, never
crosses MMQ).

## Two things I would want checked while masking QSA

- `append_qsa_cache_batch` (runtime.cpp:1462) carries the comment "the append
  sequence matches `qwen4exp_mtp_sync_cache_q1` exactly". That invariant is
  asserted in prose and checked nowhere. It indexes
  `key.data() + (row * 2U + head) * 256U`, which is correct only if the batched
  output is [rows][heads][256].
- `rotate_optional_batch` is one more q5 crossing applied to K and V after the
  per-row norm/rope loop. If the optional Hadamard rotation is present on this
  checkpoint, it is a boundary that did not exist on the q1 path in the same
  position.

Review verdict on the nine commits: no defect found by inspection. They are
consistent with each other and with the q1 paths they mirror. The risk they
carry is not a coding error, it is that each one added a q5 boundary and none
added a test across it.
