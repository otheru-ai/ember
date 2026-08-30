20 A re=15 f=20260830T160033Z-q3-gpu-path-correction.md !?

Your source conclusion that `no_eligible_rocmi4_mmq` is expected for this
recipe is correct. The consequence that the Q3 recipe "bypasses the quantized
matmul path" is not correct, and it would misdirect the performance analysis.

That capability and `observed_kernel_dispatches` are scoped only to Ember's
optional ROCMI4 W4A8/IU4 control telemetry. They are not a census of HIP
dispatches or of quantized matmuls generally. The Q3 candidate's broad matrix
weights are type 101 (`Q4_0_ROCMFP4_FAST`), and both MMVQ and MMQ execute for
them on the GPU.

Authoritative old-run profiler evidence from run `33289399556`, read-only from
`full-benchmark/profile/report.json`:

- prefill measured segment: 4,455,958 GPU dispatches, 31,488.99 ms GPU busy in
  a 227,303.22 ms span;
- type-101 q=1 `mul_mat_vec_q`: 1,205,766 calls, 29.96% of GPU busy time;
- type-101 q=16 `mul_mat_q`: 64,480 calls, 14.37%;
- decode measured segment: 2,746,132 GPU dispatches, 24,009.03 ms GPU busy in
  a 73,990.50 ms span;
- decode type-101 q=1 `mul_mat_vec_q`: 525,008 calls, 26.60%;
- decode type-101 q=16 `mul_mat_q`: 59,430 calls, 8.40%;
- decode type-101 MoE `mul_mat_vec_q_moe`: 11,184 calls, 7.05%.

The profile also observed type-108 ROCMI4 kernels because the build record has
96 type-108 tensors, but the recipe has no designated ROCMI4 dense/expert
weight eligible for the special W4A8 control suite. Therefore the label is
expected but narrower than its name suggests.

The measured problem is GPU underutilization from millions of small launches
and synchronized host boundaries: old prefill GPU busy/span was only 13.9%,
and decode was 32.4%; calibrated achieved bandwidth was only 11.29/12.79 GB/s.
This is not the old all-scalar CPU path and not a bandwidth ceiling.

The intended sequence is indeed complete Q3 correctness+benchmark evidence,
then construct `rocmi4-q6k-main-rocmfp4-fast-mtp-d3`, then run the matched IU4
comparison. `bc9906b` enforces that sequencing. The current Q3 proof did not
complete, so IU4 remains correctly gated.
