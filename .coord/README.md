# Agent coordination channel

Two agents work this repo: Codex (implements, runs GPU/CI) and Claude Code
(reviews, verifies, writes specs). The channel between them is asymmetric, so
this directory is the part that works in both directions.

## codex -> claude
Write a file into `outbox/`. Name it `<utc-timestamp>-<slug>.md`.
Claude has a persistent watcher on this directory and is notified within ~20s.
Write one when you: finish a task, hit a blocker, dispatch a workflow, or
discover something the other agent should know.

## claude -> codex
Files land in `inbox/`. Claude also pushes via `codex queue --thread <id>`,
which drains at Codex's next turn boundary.

## Why files
`codex queue` has unbounded latency when Codex is inside a long autonomous turn
(observed: two messages unread while it blocked on `gh run watch`). Codex has no
way to call Claude at all. Files work regardless of what either agent is doing,
and survive a session ending.
