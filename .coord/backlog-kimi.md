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

1. GPU-free reproduction seam. **Design work is DONE — do not redo it.**
   - proposal: `.coord/msg/20260830T171350Z-kimi-to-claude-seam-proposal.md`
   - literal-classification table (219 hits, hand-classified, incl. the
     same-value traps for 512/16/48/4/2/3):
     `.coord/msg/20260830T172506Z-kimi-to-claude-literal-table.md`
   - review + GO with 3 conditions:
     `.coord/msg/20260830T172839Z-claude-to-kimi-a.md`
   A previous session exhausted its token budget after producing the table.
   **Start from the table. Do not re-derive it.**

   Commit 1 is split so each chunk fits one session. Land them in order, each
   its own commit, each verified before the next. Rebase onto current HEAD
   first and reconcile if the literal count differs from 219.

   1a. Add `Qwen4ExpDims` to `qwen4exp_internal.h` with member defaults equal
       to today's constants, and embed it as `Qwen4ExpWeights::dims`. Change
       nothing else. Build + 90/90.
   1b. Swap the constants block `qwen4exp_runtime.cpp:18-32` to read
       `weights.dims.*`. Compiler finds every named use.
   1c. Swap the class-(a) DERIVED literals per the table — 10240, 6144, 4096,
       2048, 160, head/3, head/12, 248320. These are the ones the compiler
       cannot find; work strictly from the table's per-region rows.
   1d. Companion changes in `qwen4exp_frontier.cpp` (:1858-1861 create specs,
       :2456/:2467, :2499/:2515/:2530) and the PLE hash delegation in
       `qwen4exp_state.cpp:131`.

   Condition 3 (byte-exact `qwen4exp_*` ctest output diff) applies to the
   whole of 1a-1d taken together, not to each chunk.

2. Commit 2: the hygiene swaps (four bare `16` batch bounds ->
   `kQwen4ExpFrontierMoeMaxBatch`, `:270/:966/:1395/:1470`). Separate so
   commit 1 stays a provable no-op.

3. Commit 3: the fixture.

4. After the seam lands: an end-to-end q1-vs-batched composition test to logits
   at small dims. Must state explicitly, in a comment, that a CPU-backend F32
   fixture cannot reproduce MMVQ-vs-MMQ quantization drift — so a green result
   proves ordering/state/composition, not GPU numeric exactness.

5. Batch-width sweep test: for n in 1..16, drive each frontier subsystem
   (HC, GDN, MoE, dense) with n rows and compare against n single-row
   evaluations. Bucketing maps n to graphs of physical width 1/5/16
   (`qwen4exp_frontier_moe_cached_width`, `qwen4exp_frontier.cpp:309`), so
   n=2 and n=10 exercise zero-padded rows. Existing tests cover only a few
   specific widths. This would have caught the padded-row class of bug.

6. Test-coverage audit of the 41 engine commits since `1532d51` that had no
   independent review. For each, say what host test would fail if it were
   reverted. Commits with no such test are the coverage gaps that matter — that
   backlog is why we spent a day on the current regression. Report as a table;
   do not write the tests yet.

7. `ci/check_invariants.py` currently catches: a `src/` file in only one CMake
   list, an unregistered test, an `add_test` with no TIMEOUT, and a target
   missing from `EMBER_STRICT_TARGETS`. Propose additional mechanical
   invariants it could catch cheaply. Candidate: a HIP-only source file whose
   behaviour no host test can reach, flagged so nobody mistakes 90/90 for
   coverage of it (`quantize.cu` is exactly this case today).

8. Standing: after any codex commit touching `engine/dflash/qwen4exp/` or
   `engine/ggml/src/ggml-cuda/`, check whether existing host tests would have
   caught a regression in it. If not, file the gap. Re-check whenever HEAD
   moves.
