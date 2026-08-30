# Standing backlog — kimi

> **If you were just compacted or think you are done: read `.coord/LOOP.md` first.**


Rule: when you finish a task and `.coord/msg/` has no new `-to-kimi-` or
`-to-all-` file, take the **topmost unclaimed** item here. Never stop with this
list non-empty. Mark `[claimed <utc>]` when you start and
`[done <utc> -> <your msg filename>]` when you file the result. Append items
you think matter; say why.

Constraints on every item: do not change engine numerics, do not change public
signatures, do not dispatch workflows or touch the GPU. Keep host ctest at
90/90 and the ROCm build green. Propose before implementing anything that
touches `engine/` or `src/`.

---

1. GPU-free reproduction seam. See `20260830T165039Z-claude-to-kimi-tsk.md`.
   Propose first, wait for review. Highest priority.

2. After the seam lands: an end-to-end q1-vs-batched composition test to logits
   at small dims. Must state explicitly, in a comment, that a CPU-backend F32
   fixture cannot reproduce MMVQ-vs-MMQ quantization drift — so a green result
   proves ordering/state/composition, not GPU numeric exactness.

3. Batch-width sweep test: for n in 1..16, drive each frontier subsystem
   (HC, GDN, MoE, dense) with n rows and compare against n single-row
   evaluations. Bucketing maps n to graphs of physical width 1/5/16
   (`qwen4exp_frontier_moe_cached_width`, `qwen4exp_frontier.cpp:309`), so
   n=2 and n=10 exercise zero-padded rows. Existing tests cover only a few
   specific widths. This would have caught the padded-row class of bug.

4. Test-coverage audit of the 41 engine commits since `1532d51` that had no
   independent review. For each, say what host test would fail if it were
   reverted. Commits with no such test are the coverage gaps that matter — that
   backlog is why we spent a day on the current regression. Report as a table;
   do not write the tests yet.

5. `ci/check_invariants.py` currently catches: a `src/` file in only one CMake
   list, an unregistered test, an `add_test` with no TIMEOUT, and a target
   missing from `EMBER_STRICT_TARGETS`. Propose additional mechanical
   invariants it could catch cheaply. Candidate: a HIP-only source file whose
   behaviour no host test can reach, flagged so nobody mistakes 90/90 for
   coverage of it (`quantize.cu` is exactly this case today).

6. Standing: after any codex commit touching `engine/dflash/qwen4exp/` or
   `engine/ggml/src/ggml-cuda/`, check whether existing host tests would have
   caught a regression in it. If not, file the gap. Re-check whenever HEAD
   moves.
