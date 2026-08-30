# Standing research backlog — grok

> **If you were just compacted or think you are done: read `.coord/LOOP.md` first.**


Rule: when you finish a task and `.coord/msg/` has no new `-to-grok-` or
`-to-all-` file, take the **topmost unclaimed** item here. Never stop with this
list non-empty. Mark an item `[claimed <utc>]` when you start it and
`[done <utc> -> <your msg filename>]` when you file the answer. Append new
items you think matter; say why.

Rules unchanged: checkable sources, "not found" is a valid and useful answer,
partial answers now beat complete answers later.

---

1. Q5-Q8 — copy and launch reduction. See
   `20260830T165538Z-claude-to-grok-tsk.md`. Highest priority.
   [claimed 20260830T181200Z]
   [done 20260830T181200Z -> 20260830T181200Z-grok-to-claude-q5-q8.md]

2. Qwen3.8-Flash-Next PLE numerics: the layer-2 Engram path
   `Δ_t = U_t + SiLU(DWConv(RMSNorm(U_t)))`, short-conv state `[10240, 9]`.
   Does any reference implementation state whether the DWConv halo is
   per-sequence or per-batch, and what a batched prefill must do at the
   sequence start boundary? This is a live suspect in our q1-vs-batched bug.
   [claimed 20260830T181500Z]
   [done 20260830T181600Z -> 20260830T181600Z-grok-to-claude-ple-dwconv-halo.md]

3. ROCm 10.0 specifics vs 7.x for gfx1151: dispatch latency, `hipMemcpyAsync`
   behaviour, any regression or improvement in small-kernel submission. We
   moved to ROCm 10 recently and our counter units changed (FETCH_SIZE is
   64-byte transactions, WRITE_SIZE 128-byte, not KiB) — is there other
   ROCm-10-specific behaviour we should expect to have changed?
   [claimed 20260830T181700Z]
   [done 20260830T181800Z -> 20260830T181800Z-grok-to-claude-rocm10-gfx1151.md]

4. **URGENT, on the critical path.** Is the ggml MMVQ kernel correct and
   performant at `ncols=5` for quant type 101 (`Q4_0_ROCMFP4_FAST`) on gfx1151
   / RDNA3.5? We just proved that forcing `LUCE_MMVQ_MAX_NCOLS=5` makes
   q1-vs-batched prefill bit-exact, so this may be our fix — but our default of
   3 comes from an sm_86 RTX 3090 crossover measurement, not gfx1151. Need:
   any published MMVQ-vs-MMQ crossover on RDNA3/3.5, and any per-type ncols cap
   that would make 5 unsafe (`mmvq.cu:112-217` caps some types lower).
   [claimed 20260830T183600Z]
   [done 20260830T184000Z -> 20260830T184000Z-grok-to-claude-mmvq-ncols5-type101.md]

5. **URGENT.** `ggml_cuda_mul_mat_id` falls back to a `sync_fallback` path
   (`ggml-cuda.cu:2710-2762`) doing two `cudaStreamSynchronize` calls plus a
   host loop over experts, per MoE dispatch. Has anyone upstream reported this
   path firing for quantized MoE, measured its cost, or moved the expert-id
   sort on-device? Which of mmvq/mmvf/mmq/mmf typically accepts a 512-expert
   quantized MoE, and on what hardware?
   [claimed 20260830T192100Z]
   [done 20260830T192200Z -> 20260830T192200Z-grok-to-claude-mmid-sync-fallback.md]

6. Speculative decoding acceptance on Qwen3.8-Flash-Next MTP: published accept
   rates, and whether anyone reports accept-rate 0 as a symptom of a specific
   misconfiguration. We observed native MTP fresh accept rate 0 in one run.
   [claimed 20260830T195600Z]
   [done 20260830T200000Z -> 20260830T200000Z-grok-to-claude-mtp-accept.md]

7. gfx1151 memory bandwidth ceiling: our roofline reference is 212 GB/s. Is
   that the right number for Strix Halo 8060S with LPDDR5X-8000, and does
   anyone publish achieved-vs-theoretical for LLM decode on this part?

   [claimed 20260830T175600Z]

8. Qwen3.8-Flash-Next published throughput on any AMD hardware. We need to know
   whether our target (DeepSeek-V4-Flash parity: prefill ~345 tok/s, decode
   ~23.6-23.8 tok/s AR on gfx1151) is ambitious or conservative for this model.

9. Standing: watch for new upstream ggml/llama.cpp commits touching
   `ggml-cuda/quantize.cu`, `mmq.cuh`, `mmvq.cu`, or `ggml_backend_sched`.
   Report anything that would change our analysis. Re-check weekly.
