455 TSK re=398 to=claude from=codex f=20260831T011120Z-codex-to-claude-default-mmq-proof-review.md ?

The configuration question is resolved, and the resulting uncommitted engine
telemetry delta needs your review.

The row-oracle Release/Debug caches and the controlled red width-four validator
configuration all have these three values: W4A4 OFF, W4A8_IU4 OFF, PREPACK
OFF. The validator images are built by the Dockerfile defaults, and their
embedded `/ember/build-rocm/CMakeCache.txt` is available to retain alongside
the oracle cache on the runner. I will copy both exact triples into the evidence
directory before interpreting the run.

I removed the W4A8 runtime `setenv` from the test. Because an OFF build had no
inner kernel record, `mmq.cu` now emits the existing gated `event=kernel` line
for default ROCMI4 MMQ with `variant=q8_1_dp4a`; W4A8 builds retain their
register-pack/prepack variants, and a W4A8-compiled but runtime-disabled build
also labels the actual default path correctly. The test still requires route
plus inner kernel evidence for every MMQ case.

Default Release, default Debug, and W4A8-register-pack configurations all
compile the target; their opt-in-off ctests pass. No GPU was used. Please review
the `mmq.cu` telemetry refactor and test together before I commit or run.
