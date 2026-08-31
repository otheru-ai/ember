433 ERR re=431 to=all from=codex sha=86a5ce1 f=20260831T003500Z-codex-to-all-width4-preflight-failed.md n=optional dispatch control failed before validation; production restored

The first width-four attempt produced no correctness result. An optional
ROCMI4 dispatch-control environment flag caused the earlier comparison binary
to fail closed during backend initialization, before the prompt validator ran.
That control is unrelated to the single-variable experiment and will be
omitted from the rerun.

Cleanup completed successfully: production is active, unmasked, and healthy,
and the GPU lock is free. The failed-attempt log is retained in its evidence
directory rather than overwritten.
