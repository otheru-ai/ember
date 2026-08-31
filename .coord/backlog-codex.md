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

1. **The correctness blocker. Nothing else outranks it.** [claimed 20260830T200352Z]
   `docs/qwen3.8-performance-status.md` is the live state. Current position:
   under `LUCE_MMVQ_MAX_NCOLS=5` at `a3a50c4`, width 2 passes and widths 3, 6
   and 17 fail; every singleton mask 1/2/4/8/16 is red and mask 31 is green.
   Your own msg 216 is the key correction — mask 31 takes the whole-layer early
   return at `qwen4exp_runtime.cpp:1700-1702`, a different function from masks
   1-30, so "31 green" does not compose from the bits.
   Next: the combination sweep you proposed (7, 24, 15, 23, 27, 29, 30), and
   grok's ordering argument in msgs 285-291.

2. **Named suspect, not yet tested — the gfx1151 type-101 MMVQ [done 20260830T223313Z -> 20260830T223313Z-codex-to-all-mmvq-suspect-closed.md]
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

3. Tranche 1 — QSA rms_norm + rope into the projection graph. [done 20260830T232715Z -> 20260830T232715Z-codex-to-all-tranche1-hardware-green.md] Parameter
   mapping is closed both ways: `test_qwen_rope_graph_oracle` (`3cc509e`)
   shows both candidate mappings match the scalar reference to ~1e-7, and
   `4e972da` covers the strided-view RMS half. Run `ctest -R
   qwen_rope_graph_oracle` before and after the edit. Grok's insert spec is
   msg 229; the accounting is msgs 231/241 — the project group goes 5 → 1 on
   the shipped decode path and the **barrier count stays at 12** until the
   indexer stops reading host `index_key`.

4. Async tranche A/B. [done 20260830T233100Z -> 20260830T233100Z-codex-to-all-async-tranche-closed.md] `faa5307` measured +2.35% on the calibrated 294-token
   ABBA probe. That is consistent with the barrier census: it converts 30
   copies to async but removes **zero** barriers, and 7 of the 14 groups hold a
   single copy where async is the blocking copy it replaced. Do not expect
   more from it.

5. Standing: [done 20260830T233255Z -> 20260830T233255Z-codex-to-all-dead-path-audit.md] check `docs/dead-code-candidates.md` before counting anything.
   The Hadamard rotation stage and the QSA block scorer do not execute on the
   shipped configuration, so no A/B may be credited with removing them.

6. **Source-only physical-width branch audit.** [done 20260831T013300Z -> 20260831T013300Z-codex-to-all-width-branch-audit.md]
   With isolated ROCMI4 arithmetic green and the full-model inventory awaiting
   review/user priority, enumerate every Qwen runtime/graph branch that can
   change between the physical-width buckets other than MMVQ/MMQ arithmetic.
   Classify routing, masking, state selection, and layout predicates from
   checkable source so the next discriminator is not guessed from coincidence.

7. **Expose the adopted TV gate in hardware evidence.** [done 20260831T015100Z -> 20260831T015100Z-codex-to-all-tv-integration-landed.md]
   The user-directed criterion landed at `4cdf5c3`, but the backend validation
   report and JSON still drop its TV decision fields. Carry checked/bound,
   distance, threshold, and row index across the backend ABI and diagnostic
   output so certification records the actual acceptance reason.

8. **Guard append-only validation-report evolution.** [done 20260831T015038Z -> 20260831T015038Z-codex-to-all-abi-guard-landed.md]
   Turn the corrected TV-field placement into a C/C++ compile-time invariant:
   the new evidence block must begin immediately after the pre-existing
   `detail` tail, preserving all old member offsets.

9. **Locate the first full-graph composition divergence.** [claimed 20260831T023700Z]
   The live-type operator oracle and activation inventory trigger the isolated
   synthetic stopping rule, while the controlled full graph remains red. Add a
   default-off shadow diagnostic that advances an independent q=1 state beside
   the ordinary batched state and reports the first layer boundary whose HC
   rows differ. Require the diagnostic arm to reproduce the ordinary final
   result; a perturbation is itself evidence of hidden graph-buffer state and
   invalidates numeric localization. This is a full-graph discriminator, not a
   new synthetic kernel fixture.
