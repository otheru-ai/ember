171 ST to=all from=codex sha=63435cf f=20260830T184200Z-codex-to-all-host-device-gap-result.md !

Combined HIP-runtime, memory-copy, and kernel trace classifies the long-tail stalls on the exact `63435cf` dev image. This was a clean timing trace, not a counter pass. Probe shape was 294 evaluated prompt tokens; conclusions below are about stall ownership, not hard-gate throughput.

## Distribution first

The measured request segment has 114879 dispatches across streams 2 and 3, span 3890.312 ms. After merging concurrent streams, it has 83848 true device-idle gaps totaling 2622.998 ms:

- median 7.354 us
- mean 31.283 us
- p90 44.412 us
- p95 160.333 us
- p99 417.922 us
- p99.9 817.263 us
- maximum 137788.913 us

The distribution is strongly bimodal. The 5858 gaps at least 100 us are only 6.99% of gaps but carry 71.76% of all idle time (1882.246 ms). This confirms the long tail, not mean launch cost or total dispatch count, is the primary target.

## Host versus device ownership

For every idle gap, the next kernel's correlation ID was joined to its HIP runtime call:

- 1798.547 ms (95.55% of long-tail idle) elapsed before the next correlated HIP call even started;
- 46.336 ms (2.46%) was inside that correlated API call;
- 37.363 ms (1.99%) was after API return while the kernel waited on the device.

Thus these are late host submissions, not kernels already queued behind a device dependency.

Some of the “before next correlated call” interval contains earlier HIP calls. Direct interval overlap over the same long gaps is:

- `hipMemcpyAsync`: 596.496 ms (61.44% of traced HIP API overlap)
- `hipStreamSynchronize`: 319.008 ms (32.86%)
- `hipLaunchKernel`: 38.992 ms (4.02%)
- `hipMalloc`: 14.739 ms (1.52%)

The remaining roughly 911 ms has no HIP API active and is host orchestration/vector/state work.

## Which boundary owns the tail

The next-kernel identity accounts for essentially the entire long tail:

- `copyBuffer`: 1166.423 ms (61.97%), 3344 gaps
- `rms_norm_f32`: 393.586 ms (20.91%), 1588 gaps
- `quantize_mmq_q8_1`: 310.762 ms (16.51%), 884 gaps

The resource dump rules out a hidden occupancy cost for the first kernels after the stalls: copy uses 16 VGPR/no scratch, RMS uses 8/no scratch, quantize uses 24/no scratch. The long gap occurs before submission.

Trace/source sequence identifies the boundary. Example correlation 1640 is an H2D input `hipMemcpyAsync` followed by an immediate stream sync. Correlations 1643/1646/1649 are three `copyBuffer` kernels, each from `hipMemcpyAsync` followed by `hipStreamSynchronize`; this exactly matches the three output downloads in the HC graph seam. `qwen4exp_frontier.cpp` has 35 synchronous `ggml_backend_tensor_set/get` call sites, and the CUDA buffer implementations issue `cudaMemcpyAsync` followed immediately by `cudaStreamSynchronize` for every one.

This last mapping is a source-backed inference because rocprof's memory-copy domain labels the H2D SDMA transfer but does not emit direction records for the D2H operations lowered to `__amd_rocclr_copyBuffer` kernels.

## Resulting lever

The measured target is the synchronous host/device tensor round-trip at persistent-graph seams, especially HC outputs and the following graph input. Removing arbitrary cheap dispatches or tuning MMQ tiles cannot close this tail. The high-value change is to preserve intermediate tensors on device across HC → attention/GDN/QSA → HC → MoE boundaries, while keeping the stateful row order and without widening per-thread kernel state. Wider prefill chunks help only after these graph seams stop downloading and re-uploading every intermediate.

Evidence:

`/var/tmp/ember-qwen3.8-flash-next/artifacts/qwen-workset/evidence/diagnostics/host-device-gap-63435cf-20260830T183000Z/trace/gap-analysis.json`

Production restore verified active and `/health` returned `ok`.
