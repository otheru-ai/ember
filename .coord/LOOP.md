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

## Rules that survive compaction

- Checkable sources only. "Not found" is a valid, useful answer. Never
  approximate — a wrong lead costs a GPU run and a production outage.
- Partial results now beat complete results later.
- GPU time and production downtime are authorized (user decision) — but only
  `codex` touches the GPU, runner, or workflows.
- `codex` implements. `claude` reviews and verifies independently. `grok`
  researches and cites. `kimi` owns host-side reproducibility. No agent counts
  its own review as independent review.
