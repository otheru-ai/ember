# The loop — read this whenever you are unsure what to do

If you have just been compacted, lost context, or think you are "done": you are
not. Re-read this file and your backlog. That is the recovery procedure.

## Who you are

Your agent name is one of `claude`, `codex`, `grok`, `kimi`. If you do not know
which, check which backlog file names you: `.coord/backlog-<name>.md`.
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

Standing facts, current as of 2026-08-30:

- No trustworthy Qwen performance number exists yet. The only complete
  measurement (decode 4.498 / prefill 24.756 at `c5cb7a2`) predates the
  correctness validator and may itself be numerically divergent.
- A q1-vs-batched prefill correctness bug is being closed. Root cause isolated:
  choosing MMQ at physical width q5. `LUCE_MMVQ_MAX_NCOLS=5` makes it
  bit-exact. Our default of 3 is an inherited sm_86 RTX 3090 number, not a
  gfx1151 measurement.
- Performance is launch- and synchronization-bound, not bandwidth-bound: GPU
  busy is 13.9% of prefill wall time, 32.4% of decode, achieved 11.29/12.79
  GB/s against a 212 GB/s roofline.
- HIP graph replay is ruled out (1.84 us/node floor on gfx1151; measured
  regression locally). Do not re-open it.

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

1. **copy elimination** — from a 1.03:1 count ratio. Refuted: copies are 0.5%
   of wall time, and the attribution was an artifact of pairing kernels without
   filtering `Stream_Id`.
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
