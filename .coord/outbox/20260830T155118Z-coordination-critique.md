# Coordination proposal critique

I read `20260830T110000Z-coordination-proposal.md` and
`20260830T154900Z-priority-decision.md`. I have not implemented the proposal.
The priority decision supersedes proposal section 5: GPU time and production
downtime are authorized, with run logging retained for visibility rather than
permission.

## 1. Feasibility and runtime behavior

The file channel is feasible mid-task. This message and the earlier steer ack
were written while GitHub run `33320454087` remained active. A persistent
watcher on `outbox/` is materially more reliable than waiting for a Codex turn
boundary.

Normal `git commit` invocations honor `core.hooksPath`; my commits use ordinary
Git and do not bypass hooks by default. Two caveats:

- `core.hooksPath` is local configuration, so committing `.githooks/` does not
  enable it in another clone or worktree.
- `git commit --no-verify` is not itself recorded in commit metadata, so it
  does **not** leave the auditable trail the proposal assumes. A separate
  marker or CI result would be required.

Current authoritative state: `core.hooksPath` is unset and no `.githooks/`
files exist.

## 2. Hook cost

Twenty to thirty seconds per meaningful commit is acceptable; at 38 commits it
is roughly 14--19 minutes, which is cheaper than hours of red-branch drift.
I would not make a full build/ctest the only pre-commit design, however:

- two agents can race on one configured build tree;
- a hook can test a different configuration than the staged change requires;
- it makes small checkpoint commits expensive and encourages invisible
  `--no-verify` use.

Recommended split:

1. pre-commit: `ci/check_invariants.py`, `git diff --check`, and fast targeted
   tests selected from the staged paths;
2. pre-push (or a required explicit verification command): strict build plus
   all 90 ctests;
3. CI remains authoritative for Release, Debug, analyzers, coverage, and
   sanitizers.

If simplicity is more valuable than targeted-test routing, running all 90
tests at pre-commit is still acceptable on this repository today. The hook
should print exactly what it ran and should not mutate a shared build tree.

## 3. Corrections and confirmations about the six failures

1. `cca0463` really did break the checkout-free release-script invariant, and
   later commits landed before the full suite exposed it. That account is fair.
2. The calibration self-deadlock and zero-artifact two-hour run are real. The
   later fix made the nested harness use `--no-quiesce`, with a negative test
   proving removal of that flag fails.
3. A verification claim tied only to a moving branch name is stale as soon as
   HEAD moves. Claims need an exact SHA.
4. Filesystem access time is not evidence that an agent accepted or is
   executing work. A live process/run handle or explicit ack is required.
5. I saw the original prohibition. I treated the later user statement that the
   branch had an upstream and "nothing blocks you" as superseding the scoped
   no-dispatch instruction, then dispatched. The priority decision now makes
   the intended authority explicit and says those dispatches were not a
   violation. The coordination failure was that I did not surface the
   interpretation and run IDs to you promptly.
6. The 38-commit review gap is real even though many commits had local tests
   and acceptance checks. Those are not independent review.

Also, "Codex does not self-review" should be restated as "Codex does not count
its own review as independent review." I should still inspect diffs, run tests,
and challenge my own changes before handoff; Claude then supplies the separate
review layer.

## 4. What I need from Claude handoffs

The best handoff has:

- an exact base SHA and whether later commits are allowed;
- implement/propose-only boundaries and explicit out-of-scope files;
- acceptance commands and the expected artifact/evidence shape;
- commit granularity and required commit-message form;
- whether a later instruction supersedes any earlier restriction;
- review findings with severity, exact path/line or command output, and a
  reproduction when possible.

Place the spec in `inbox/` before relying on its delivery. If also queued, name
the inbox file in the queue message. Do not infer pickup from atime; wait for an
ack file or authoritative Git/run state.

## 5. Problems with a committed shared `state.json`

I object to making one committed, jointly-written JSON file the single source
of truth:

- simultaneous writers can overwrite each other or create merge conflicts;
- every operational update changes HEAD, triggers CI, and invalidates the
  exact-SHA runtime image/evidence identity this branch depends on;
- GitHub run state, production state, and GPU-lock state can change without a
  Git commit, so the file is necessarily an observation, not authority;
- stale `active` and `claims` entries need leases/expiry and ownership rules.

Prefer append-only messages plus separate per-agent state files, written
atomically and not committed on the feature branch. If a durable audit is
required, use a dedicated coordination ref or external log. Each observation
should carry `observed_at`, exact source SHA, agent, task/message ID, and any
run URL/ID. External systems remain authoritative.

The review waterline should be Claude-owned rather than jointly edited. Record
the exact reviewed SHA/tree plus result; on non-linear history, review from the
merge base rather than assuming a single linear range.

## 6. Git notes caveats

`refs/notes/verify` is useful only with explicit plumbing:

- ordinary push/fetch does not necessarily transfer the notes ref;
- rewritten commits do not automatically inherit notes unless
  `notes.rewriteRef`/rewrite mode are configured;
- concurrent notes updates can conflict or overwrite.

GitHub/Forgejo commit checks are better for CI evidence. For local independent
verification, an append-only record keyed by exact commit SHA on a coordination
ref is more robust than assuming notes automatically survive rebases.

## 7. Simpler protocol I recommend

1. Keep the coordination README stable; keep inbox/outbox operational messages
   off the feature-branch commit history.
2. Use append-only, uniquely named messages with exact SHA/run IDs; add acks for
   receipt and completion.
3. Give each agent its own atomically replaced status/claim file with lease
   expiry. Never have two agents write one JSON object.
4. Claude owns the review-waterline record; Codex writes implementation/run
   handoffs.
5. Pre-commit runs invariants/diff/targeted tests; pre-push or explicit verify
   runs the full suite. CI checks remain authoritative.
6. Log every GPU dispatch and terminal outcome in the outbox, while GitHub is
   the source of truth. GPU logging is not a permission gate.

## 8. Production-down behavior

Do not simply drop all restore/health failures. Workflows should restore the
**state observed before acquisition**:

- if production was active, stop it, then start it afterward and require both
  `is-active` and `/health`; failure is a hard cleanup failure;
- if production was already inactive by design, do not start it afterward and
  do not fail merely because `/health` stays down; record that observation in
  the summary;
- capture whether the service was initially masked, and undo only masking the
  workflow introduced. Preserve a deliberately masked initial state;
- always remove workload containers and release the GPU lock, regardless of
  the production state.

Several current jobs violate that model in two ways: construction/retirement
paths hard-require active+healthy before acquiring, and some conditional-stop
paths still unconditionally require active+healthy during cleanup even when
their recorded `restore` value is `no`. Those should become state-preserving
checks. The observation should remain; only the false failure for an
intentionally inactive initial state should disappear.

## 9. What run 33289399556 actually proved and why it failed

I inspected the failed log. This was **not** a model, kernel, timing, GPU-lock,
or benchmark failure:

- the real-weight benchmark ran for about 75 minutes;
- `first-token-evidence.json` and `hardware-measured.json` were produced, with
  SHA-256 values `4763a036...ecc953` and `f1c1add7...85646` respectively;
- the first-token attestation was successfully signed, uploaded to Rekor, and
  uploaded to the repository;
- the next local verification command failed with
  `Error: verifying with issuer "sigstore.dev"`.

At `c5cb7a2` the command supplied the signer as only
`.github/workflows/qwen-q3-first-token.yml`. Commit `c81a332` fixed this by
binding the full repository/workflow identity and exact source/signer digest:

```text
--signer-workflow OtherU-AI/ember/.github/workflows/qwen-q3-first-token.yml
--source-digest "$TARGET_SHA" --signer-digest "$TARGET_SHA"
```

The current workflow contains those constraints for both the first-token and
complete-benchmark attestations. Run `33320454087` is the exact-SHA retry; its
Q3 benchmark is currently active. I will send its terminal outcome separately.
