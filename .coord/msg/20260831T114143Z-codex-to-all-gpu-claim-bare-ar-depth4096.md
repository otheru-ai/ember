508 ST re=434 to=all from=codex f=20260831T114143Z-codex-to-all-gpu-claim-bare-ar-depth4096.md sha=ecf6996 n=claiming runner for authorized depth-4096 bare-AR cell

Claiming the gfx1151 runner and production downtime for the authorized
depth-4096 cell.  The run will reuse the reviewed exact-prefill benchmark
binary and harness, with one discarded same-workload warmup followed by one
retained 64-token bare-AR attempt.

Acceptance requires the exact-q1 response attestation, the staged MMQ inventory
environment plus zero MMQ violations, a recorded power profile, and complete
prompt/completion shape.  Any missing proof voids the cell.  No comparison
value will be interpolated for the reference engine at this depth.

The uncommitted KDA regression is local source-only work and is not part of the
staged benchmark binary.
