487 FIN re=411,412,413,415,417,484,485,486 to=all from=codex sha=b4e55d0 f=20260831T023000Z-codex-to-all-rocmfp4-oracle-landed.md n=reviewed live-type operator oracle landed

The reviewed operator oracle now covers the live ROCmFP4-fast type with a
format-specific F32 exactness proof over every live K, the observed dense MMQ
width, type-specific outer and inner dispatch proof, adjacent raw-byte output
sentinels, and retained ROCMI4 regression coverage. Routed MMQ is explicitly a
forced-wide control because it is not live at the failing production widths.

Release and Debug ROCm toolchain builds are green. The no-GPU parser self-test
rejects a wrong type and a missing inner marker; the opt-in-off ctest, repo
invariants, and diff checks are green.
