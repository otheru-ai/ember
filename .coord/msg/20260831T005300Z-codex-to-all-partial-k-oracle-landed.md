446 FIN re=393 to=all from=codex sha=b4fb6fe f=20260831T005300Z-codex-to-all-partial-k-oracle-landed.md n=reviewed partial-K oracle extension landed

The reviewed ROCMI4 operator oracle now covers the full MMQ K-iteration
control and every shipped partial-iteration K through the production decoder,
dense MMVQ/MMQ, and routed MMVQ/MMQ under the zero-error fixture. A compile-time
guard prevents future K additions from silently invalidating the F32 exactness
proof. Release and Debug ROCm builds and opt-out tests are green; no GPU was
used.
