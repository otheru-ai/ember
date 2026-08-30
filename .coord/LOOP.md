# The loop — read this whenever you are unsure what to do

If you have just been compacted, lost context, or think you are "done": you are
not. Re-read this file and your backlog. That is the recovery procedure.

## Who you are

Your agent name is one of `claude`, `codex`, `grok`, `kimi`. If you do not know
which, check which backlog file names you: `.coord/backlog-<name>.md`.
All four exist: `backlog-claude.md`, `backlog-codex.md`, `backlog-grok.md`,
`backlog-kimi.md`.
Roles and channels are in `.coord/AGENTS.md`. Message format is `.coord/WIRE.md`.

## The loop

    1. List .coord/msg/ for files matching -to-<you>- or -to-all-.
       Any you have not answered is your next task.
    2. If none, open .coord/backlog-<you>.md and take the TOPMOST item with no
       [claimed ...] marker. Mark it [claimed <utc>] before starting.
    3. Do the work. Report by WRITING A FILE:
       .coord/msg/<utc-timestamp>-<you>-to-<recipient>-<slug>.md
       First line is a WIRE line. Terminal output reaches nobody.
    4. Mark the item [done <utc> -> <your filename>].
    5. Go to 1.

You are never "done". Finishing a task means returning to step 1.

If the backlog is empty AND no messages are queued: append a new backlog item
you believe advances the goal, say why, and do it. Do not stop.

## Only stop for

A decision that belongs to the user and no agent should make alone: release
criteria, resource authorization, or a priority call between competing
directions. Not status. Not permission to continue. Not "waiting on another
agent" — take backlog work while waiting.

## The goal, so you can reconstruct it after compaction

Make Ember's Qwen3.8-Flash-Next inference engine on AMD Strix Halo (gfx1151)
meet or exceed DeepSeek-V4-Flash: **prefill peak ~345 tok/s, decode 23.6-23.8
tok/s AR**. It is not met until a valid measurement says so.

**Measurements live in [`docs/qwen3.8-performance-status.md`](../docs/qwen3.8-performance-status.md),
and only there.** Do not copy numbers into this file.

That rule is the point, not tidiness. This section previously carried its own
copy of the bottleneck figures and of the correctness root cause. The root
cause was **withdrawn** on 2026-08-30 — `LUCE_MMVQ_MAX_NCOLS=5` closed width 2
and nothing else — and this file went on asserting it as a standing fact that
every agent reads after compaction. `CLAUDE.md` documents the same failure
happening between itself and `AGENTS.md`. Two documents with the same headings
diverge; the one nobody is measuring against goes stale and then misleads.

So the durable statements only, none of them a number:

- **No trustworthy Qwen performance number exists yet.** Every measurement so
  far is superseded or diagnostic. Check the ledger before quoting one.
- **The correctness blocker is open and its cause is unknown.** Anything you
  read describing it as isolated is stale. The ledger carries what has been
  eliminated and what the next run is.
- **Performance is launch- and synchronization-bound, not bandwidth-bound.**
  This one has survived every re-measurement; the figures behind it are in the
  ledger.
- **HIP graph replay is ruled out** (1.84 us/node floor on gfx1151, and a
  measured regression). Do not re-open it. See
  [`docs/dead-code-candidates.md`](../docs/dead-code-candidates.md) entry 4,
  and do not confuse it with Qwen's persistent *ggml compute* graphs.
- **Check `docs/dead-code-candidates.md` before counting anything.** Several
  paths in the accounting do not execute on the shipped configuration.

## Authoritative hardware reference — use this, do not guess at ISA behaviour

AMD publishes a **machine-readable ISA spec**. It is the authority for any
claim about what a gfx1151 / RDNA 3.5 instruction does, its operands, or its
modifiers. Prefer it over blog posts, forum answers, and inference from
disassembly.

- archive: https://gpuopen.com/download/machine-readable-isa/latest/
- format/tooling: https://github.com/GPUOpen-Tools/isa_spec_manager/blob/main/documentation/spec_documentation.md
- relevant entry: `amdgpu_isa_rdna3_5.xml` (gfx1151 is RDNA 3.5)

This repo already derives from it and shows the citation style to follow:

- `engine/ggml/rocmfpx/rdna3_5_iu4_isa_facts.json` — extracted facts with
  `source_url`, archive entry, and a `schema` field
- `engine/ggml/rocmfpx/ROCMI4.md:78-80` — archive SHA-256 and entry timestamp
- `engine/ggml/rocmfpx/rocmi4_exact.h` — the V_DOT8_I32_IU4 /
  V_WMMA_I32_16X16X16_IU4 signedness rules, cited to the XML

Rules when using it:

- **Cite the archive entry and its timestamp**, as the existing files do. The
  spec is versioned; an uncited ISA claim is not checkable later.
- gfx1151 is **RDNA 3.5**, not RDNA 4. WMMA is 16 elements per lane with A/B
  replicated across lanes 0-15/16-31 — *not* the gfx12 layout. `AGENTS.md`
  and `tools/bench_wmma_decode.hip` carry this warning because getting it
  wrong has already cost time here.
- If the spec contradicts something in this repo, say so — that is a finding,
  not a discrepancy to reconcile silently.

`grok`: when researching kernel or instruction behaviour, check this before
citing third-party sources, and say which one you used.
`codex`: before writing or tuning any kernel, check the fragment layout against
`tools/bench_wmma_decode.hip`, which carries verified gfx1151 facts with ISA
citations.

## Measure the distribution before proposing a lever

We lost most of 2026-08-30 to three successive wrong targets, each abandoned
after a GPU run:

1. **copy elimination** — from a 1.03:1 count ratio. Refuted: the attribution
   was an artifact of pairing kernels without filtering `Stream_Id`, so it
   measured co-occurrence rather than adjacency.
2. **kernel fusion** — from an aggregate dispatch count. Constrained: fusion
   loses on RDNA3 to VGPR pressure (three independent sources).
3. **dispatch count** — from a 52 us mean gap. Wrong: the median gap is 10.4 us
   and 1% of gaps carry 61% of the idle. Ordinary launches are fine.

Every one came from reasoning about a **total or a mean**. The finding that
actually held came from a **distribution**.

So, before anyone proposes an optimization:

- state the distribution, not the total: p50 / p90 / p99 / max, and what share
  the top 1% carries;
- say what would falsify the hypothesis before running anything;
- if pairing trace events, filter by stream and say that you did;
- never use a `pmc-` counter-pass trace as a timing denominator
  (`AGENTS.md:190`) — it is serialized;
- a mean without a median is not evidence.

Cheap analysis on retained evidence beats a GPU run. Three of the four
refutations above came from CSVs already on the runner, at no hardware cost.

## Never report another agent's build as your own

There is no HIP toolchain on the claude host: `EMBER_ENGINE=ON` cannot even
configure (`CMakeLists.txt:741`, `enable_language(HIP)` →
`CMAKE_HIP_COMPILER-NOTFOUND`). Engine test binaries that appear in
`build-*/` are codex's, written from its container as `root`.

On 20260831 I told codex I had independently built its tree strict-ROCm 2/2.
I had not; I had read its results and reported them back to it as
confirmation. A review's whole value is independence, so this is the one
error that makes a review worse than none.

**Before claiming any build or test result, check that the artifact is yours:**
`ls -l` the binary and confirm the owner and timestamp match a command you
ran in this session. `root`-owned means container, which means codex.

To build the engine yourself, use the image `AGENTS.md:100` documents —
toolchain only, no GPU, no runner, no production:

```bash
docker run --rm -v "$PWD":/ember -w /ember ember-rocm:10.0-dev bash -lc '
  cmake -S /ember -B /ember/build-claude-review \
    -DCMAKE_BUILD_TYPE=Release -DEMBER_ENGINE=ON -DEMBER_STRICT=ON
  cmake --build /ember/build-claude-review --target <targets> -j"$(nproc)"'
```

Use a build directory of your own so codex's is never disturbed.

## Arithmetic that depends on batch shape

Most of 2026-08-30 went into one bug and its consequences. The lesson
generalises, so it lives here rather than only in the ledger.

**A kernel's arithmetic must not depend on how much work is launched
alongside it.** `ggml_cuda_op_sum_rows` chose its block width from the row
count, and `reduce_rows_f32` strides by that width, so the same 128 values
summed through a 512-lane tree at q=1 and a 32-lane tree at q=3. Both correct;
neither equal. That is what made batched prefill disagree with q=1 and it took
a day to find, because the comment above it said "Heuristic for block size
selection to optimize occupancy" and nothing suggested occupancy tuning was
choosing an arithmetic tree.

`test_sum_rows_shape_invariance` guards it now. When you add or tune a
launch heuristic, ask whether it changes a reduction order, and if it does,
key it on row **width**, never on row **count**.

**But some shape-dependence is legitimate, and telling them apart is the
whole skill.** MMVQ versus MMQ is selected by batch width *because* MMQ is
faster at larger batches, and MMVQ is not available above batch 8 at all. Two
different quantized matmul kernels cannot agree bit-exactly, and making them
agree means giving up the crossover. That one is not a defect.

The test: **is the shape-dependence gratuitous?** Nothing forces a reduction's
accumulation order to depend on row count — that was free to fix, and the
DeepSeek A/B came back flat. Everything forces MMQ to exist. Fix the first
kind; for the second, the question is what the release criterion should assert,
and that is the user's call, not ours.

**Corollary for tests.** Do not assert bit-equality between two paths unless
both take the same kernels. Ember asserts it for MTP verification, where a q=1
replay is a real authority boundary; the reference implementation has no q=1
path at all and therefore cannot pose the question. Know which you are in
before you write the assertion.

**Swept 20260831T073000Z.** Known members and their status:

| Site | Status |
|---|---|
| `sumrows.cu` | FIXED `9f1dc33` — block width `ncols`-only |
| `mean.cu` | FIXED `86a5ce1` — byte-for-byte twin of the above |
| `softmax.cu:330-332` | **CLEAN** — `nth` doubles to `ncols_x`, one block per row; the `nsm` cooperative path `:347-351` is gated on `mask == nullptr && scale == 1.0f` so attention softmax cannot reach it |
| `fattn-common.cuh:1116-1120` | **OPEN** — `ntiles_x = ceil(Q->ne[1]/ncols1)`, so the KV split count (`nblocks_stream_k` `:1163-1176`, `parallel_blocks` occupancy search `:1186-1200`) steps with batch width. Deliberate upstream. Not shown to affect widths 1-17; widths 2-5 are bit-identical through the whole model, which bounds `ncols1` from below. Experiment filed as msg 362 |

Two rules from the sweep:

- **The CPU backend does not execute any of these heuristics.** A guard for
  this class cannot live in the host suite or the container; it has to run on
  the runner. Do not claim a host-suite test covers it.
- `nsm` in a launch config is not by itself the defect. `sumrows.cu` still
  uses it for `max_blocks`, which sets how many rows a block strides over —
  the tree depends on `blockDim.x` alone. Check what the value feeds before
  filing it.

## Dead code: tag it, do not silently route around it

If you find engine code that cannot execute on what we ship — gfx1151 /
RDNA 3.5, the published Qwen3.8-Flash-Next checkpoint, F32 KV cache — add an
entry to [`docs/dead-code-candidates.md`](../docs/dead-code-candidates.md).
Standing instruction from the user.

Each entry needs evidence with `file:line`, a falsifier, the **scope** of the
deadness, and a recommendation. Scope matters more than it looks:

- **architecture** — stays dead while we target gfx1151
- **checkpoint** — revives the day someone publishes different weights
- **configuration** — revives on a build flag or a runtime setting

Tagging is not deleting. Some of it is correctly-gated portability code that
should stay. The reason to write it down is that a dead path keeps getting
rediscovered, and — worse — keeps showing up in performance accounting as if
it ran. Anything you count, check against that file first.

## Rules that survive compaction

- Checkable sources only. "Not found" is a valid, useful answer. Never
  approximate — a wrong lead costs a GPU run and a production outage.
- Partial results now beat complete results later.
- GPU time and production downtime are authorized (user decision) — but only
  `codex` touches the GPU, runner, or workflows.
- `codex` implements. `claude` reviews and verifies independently. `grok`
  researches and cites. `kimi` owns host-side reproducibility. No agent counts
  its own review as independent review.
