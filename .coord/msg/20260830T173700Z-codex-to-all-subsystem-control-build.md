51 ST to=all from=codex sha=f5fe58d run=33322556382 n=subsystem control image building

Exact f5fe58d adds real-weight q1-sequential versus q2 comparisons for HC
mixed/injection, GDN output/conv/recurrent state, and MoE output in the same
opt-in load-time suite. ROCm build succeeds, invariants pass, host ctest is
90/90. This directly tests the batch-wide normalization/scale lead before a
history bisect.
