# Standing backlog — codex

> **If you were just compacted, restarted, or think you are done: read
> `.coord/LOOP.md` first.**

Rule: when a message is answered and `.coord/msg/` has no unanswered
`-to-codex-` or `-to-all-`, take the topmost unclaimed item. Never idle while
this list is non-empty. Mark `[claimed <utc>]` / `[done <utc> -> ref]`.

You implement and you own the hardware-facing evidence. You are the only agent
that touches the GPU, the runner, or workflows. GPU time and production
downtime are authorized by the user, but **announce the claim in `.coord/msg/`
before taking the box**, and report the release. Report every result by writing
a file into `.coord/msg/` — terminal output reaches nobody.

Do not publish any performance number while the correctness blocker is open.
`claude` reviews your commits; you do not self-review.

---

1. **The correctness blocker. Nothing else outranks it.**
   `docs/qwen3.8-performance-status.md` is the live state. Current position:
   under `LUCE_MMVQ_MAX_NCOLS=5` at `a3a50c4`, width 2 passes and widths 3, 6
   and 17 fail; every singleton mask 1/2/4/8/16 is red and mask 31 is green.
   Your own msg 216 is the key correction — mask 31 takes the whole-layer early
   return at `qwen4exp_runtime.cpp:1700-1702`, a different function from masks
   1-30, so "31 green" does not compose from the bits.
   Next: the combination sweep you proposed (7, 24, 15, 23, 27, 29, 30), and
   grok's ordering argument in msgs 285-291.

2. **Named suspect, not yet tested — the gfx1151 type-101 MMVQ
   specializations.** `engine/ggml/src/ggml-cuda/mmvq.cu:1495-1516` selects a
   *different kernel* by `ncols_dst`: 1 → `mul_mat_vec_rocmfp4_unroll2_launch`,
   4 → `mul_mat_vec_rocmfp4_4col_reuse_launch`, otherwise generic. Their
   bit-exactness is asserted only in a comment.
   Consequence: **q=1 runs `unroll2` and batched runs generic**, and the
   differential compares batched against q=1 — so if those disagree, the
   reference itself may be the wrong side, and "mask 31 green" may only mean
   "unroll2 agrees with unroll2".
   Falsifier: add an env guard (the neighbouring code already uses
   `DFLASH_CUDA_MMVQ_*`) that forces the generic path, then compare q=1 with
   and against the specialization. If they differ, the comment is false.
   Residual that does not fit yet: widths 2 and 3 both map to physical 5, so
   any kernel-selection story still has to explain why 2 passes.

3. Tranche 1 — QSA rms_norm + rope into the projection graph. Parameter
   mapping is closed both ways: `test_qwen_rope_graph_oracle` (`3cc509e`)
   shows both candidate mappings match the scalar reference to ~1e-7, and
   `4e972da` covers the strided-view RMS half. Run `ctest -R
   qwen_rope_graph_oracle` before and after the edit. Grok's insert spec is
   msg 229; the accounting is msgs 231/241 — the project group goes 5 → 1 on
   the shipped decode path and the **barrier count stays at 12** until the
   indexer stops reading host `index_key`.

4. Async tranche A/B. `faa5307` measured +2.35% on the calibrated 294-token
   ABBA probe. That is consistent with the barrier census: it converts 30
   copies to async but removes **zero** barriers, and 7 of the 14 groups hold a
   single copy where async is the blocking copy it replaced. Do not expect
   more from it.

5. Standing: check `docs/dead-code-candidates.md` before counting anything.
   The Hadamard rotation stage and the QSA block scorer do not execute on the
   shipped configuration, so no A/B may be credited with removing them.
