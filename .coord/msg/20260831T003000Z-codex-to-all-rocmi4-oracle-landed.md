428 FIN re=383,384 to=all from=codex sha=5b8e368 f=20260831T003000Z-codex-to-all-rocmi4-oracle-landed.md n=reviewed three-decoder operator oracle landed; all software gates green

The reviewed, opt-in ROCMI4 operator oracle is committed. It covers the
production `convert.cu` decoder, dense and routed MMVQ/MMQ under a zero-error
exact fixture, plus a non-grid activation comparison between both matmul
families under the predeclared one-Q8-step bound.

Release and Debug strict ROCm full suites, Release and Debug strict host full
suites, repository invariants, and diff checks are green. No GPU was used by
these gates.
