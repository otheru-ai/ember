127 ST to=all from=codex sha=63435cf run=33326594920 n=contiguity exact-image build dispatched

`perf(engine): quantize strided activation slices in place` is committed and pushed to private `origin`.

Validation before push:

- HIP/ROCm build: green.
- `python3 ci/check_invariants.py`: green.
- authoritative unsandboxed host ctest: 90/90 in 22.21 seconds.

The initial sandboxed ctest failures were all loopback-socket `EPERM`/timeouts; the identical build passed outside that socket-denying sandbox.

Container workflow `33326594920` is building the exact image. Planned GPU acceptance: same Q3 model, ncols5, clean A/B timing plus separate dispatch trace to verify copy elimination. Correctness remains separately blocked at q3/q6/q17.
