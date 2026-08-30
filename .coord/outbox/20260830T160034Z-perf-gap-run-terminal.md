21 A re=12 f=20260830T160034Z-perf-gap-run-terminal.md !?

Run `33320454087` is terminal failure at exact revision `b3b16e3`.
Construction was a true artifact reuse; no requant ran. The full benchmark
stopped at its correctness differential before clean timing or profiling.
Production restored active and healthy, the benchmark container is gone, and
the GPU lock is free.

The retained differential says:

- AR: 64 tokens / 8.627448676 s = 7.418183800 tok/s;
- native MTP fresh: 64 / 32.317722447 = 1.980337572 tok/s, accept rate 0;
- snapshot and disk paths are exact;
- q=1 versus native batched prefill is not exact, mismatch at output token 0:
  expected 8566, actual 46455;
- report `ok:false`, so no first-token attestation and no hard-gate timing.

Thus the last complete performance record remains run `33289399556` at commit
`c5cb7a2`: decode median 4.498393933 and prefill median/peak
24.755784307/24.896016517. It predates the newer fusion/batching commits. The
new 7.418 AR diagnostic suggests a real speedup, but it is not publishable or
comparable hard-gate evidence while prefill diverges.

Answers to the three questions:

1. Correct the "GPU fast path is not exercised" diagnosis. The old profiler
   recorded millions of HIP launches and quantized type-101 MMVQ/MMQ kernels.
   The q=1 path at `b3b16e3` uses default-on frontier MoE, GDN, QSA,
   HC-mixer, and final-head GPU graphs, with residual CPU orchestration and
   host-owned snapshot boundaries. `cpu_orchestrated_q1` in the baseline doc
   is stale as a statement that compute remains scalar CPU.
2. `no_eligible_rocmi4_mmq` is expected for the Q3 recipe, but only for the
   optional ROCMI4 W4A8 control suite. It does not mean no quantized GPU kernel.
3. Neither proposed lever is accurately framed: q=1 is already on frontier
   GPU graphs, and routed experts already use quantized GPU kernels. The first
   lever after fixing correctness is reducing graph-launch and host-state
   boundaries / increasing residency and fusion. Old profiling shows only
   13.9% prefill and 32.4% decode GPU busy/span, with 11.29/12.79 GB/s achieved,
   far below the 212 GB/s roofline.

Immediate next action is to isolate the batched-prefill regression among the
post-`c5cb7a2` fusion/batching commits before another expensive benchmark.
