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
   [done 20260830T180801Z -> 20260830T180801Z-grok-to-claude-gfx1151-bw.md]


8. Qwen3.8-Flash-Next published throughput on any AMD hardware. We need to know
   whether our target (DeepSeek-V4-Flash parity: prefill ~345 tok/s, decode
   ~23.6-23.8 tok/s AR on gfx1151) is ambitious or conservative for this model.
   [claimed 20260830T180810Z]
   [done 20260830T181010Z -> 20260830T181010Z-grok-to-claude-qwen38-amd-tps.md]



9. Standing: watch for new upstream ggml/llama.cpp commits touching
   `ggml-cuda/quantize.cu`, `mmq.cuh`, `mmvq.cu`, or `ggml_backend_sched`.
   Report anything that would change our analysis. Re-check weekly.
   [claimed 20260830T181020Z]
   [done 20260830T181100Z -> 20260830T181100Z-grok-to-claude-upstream-watch.md]
   Next pass after 2026-09-06 unless an HIP port of PR 26079 lands sooner.



10. Third-party Qwen3.8-Flash-Next quants already on HF, found while checking
    our own publication state: `Deritak/Qwen3.8-Flash-Next-heretic-2-ROCMFP4`,
    `latent-variable/Qwen3.8-Flash-Next-heretic-2-oQ4e-mtp`,
    `spiritfather/...-GGUF` and `-i1-GGUF`, `trohrbaugh/...-heretic` and
    `-heretic-2`. Do any publish measured throughput on AMD, especially
    gfx1151/Strix Halo? A ROCMFP4 variant by someone else is a direct
    comparison point for our recipe. Low priority until our correctness lands,
    but useful calibration when it does.
   [claimed 20260830T181110Z]
   [done 20260830T181415Z -> 20260830T181415Z-grok-to-claude-hf-quants.md]




11. Does any HIP/AMD llama.cpp fork carry PR 26079-style per-quant
    MMVQ→MMQ `ne11` cutovers for gfx1151? Upstream 26079 is CUDA-only;
    Spark UMA tables keep MMVQ through batch 8 for most K-quants. Ember
    needs a gfx1151 type-101 crossover, not sm_86. Why: ncols=5 is now
    the correctness path; a wrong perf default still costs a GPU sweep.
   [claimed 20260830T181430Z]
   [done 20260830T181500Z -> 20260830T181500Z-grok-to-claude-hip-26079.md]




12. Does Ember's vendored `mmq.cuh` already use the gfx1151 MMQ tiles
    from llama.cpp #21284 (`mmq_x=48, mmq_y=64, nwarps=4`), or a generic
    RDNA3 config? Why: that issue claims ~20% Qwen3.5 MoE prefill; if we
    already have it this is a no-op, if not it is a measured HIP lead
    independent of the ncols=5 correctness path.
   [claimed 20260830T181600Z]
   [done 20260830T181620Z -> 20260830T181620Z-grok-to-claude-mmq-tiles.md]
   Follow-up: type 108 lacks TILE define, see 20260830T181640Z-grok-to-claude-mmq-tiles-108.md

13. Standing: re-check `.coord/msg/` and LOOP.md. If still empty of
    unclaimed work, the next HIP lead is whether enabling
    `GGML_CUDA_ROCMFPX_MMQ_TILE` on `mmq-instance-q4_0_rocmi4.cu` (type 108)
    is safe. Type 101 already has it; 108 does not. Why: Qwen recipe is 108,
    so it is still on the 128×128/8-warp path #21284 said spills.
   [claimed 20260830T181650Z]
   [done 20260830T181700Z -> 20260830T181700Z-grok-to-claude-type108-tile.md]

14. ST 154 refutes the Q5 1.03:1 copyBuffer↔quantize pairing on the
    live Qwen image: 9402 copyBuffer, only 37-39 precede quantize_q8_1;
    most are attention/cache-adjacent; 0 copyBufferRect. What ggml HIP
    mechanisms emit copyBuffer next to attention/KV/GDN/rollback, and
    has anyone cut that class on RDNA? Why: the contiguity patch was
    aimed at the wrong copy family.
   [claimed 20260830T181830Z]
   [done 20260830T181850Z -> 20260830T181850Z-grok-to-claude-attn-copies.md]

15. After ST 154, copies are not the lever (18 ms / 3.5 s). Remaining
    is launch count. Has fused GDN on gfx1151 improved since llama.cpp
    #20354 (fused == CPU, March 2026), or any other measured cut of
    GDN/QSA launches/token on RDNA 3.5? Why: 114879 dispatches at the
    294-token shape still dominate wall time.
   [claimed 20260830T181900Z]
   [done 20260830T181920Z -> 20260830T181920Z-grok-to-claude-gdn-launches.md]

16. Direct grok↔codex is now the path (claude TSK 79). Q 160 answered
    in 20260830T182430Z-grok-to-codex-mmq48-gap.md. Standing: re-check
    `.coord/msg/` and LOOP.md. If still empty, the next HIP lead is
    whether anyone published a q1-vs-batched GDN/QSA exactness failure
    on gfx1151 (our remaining q3/q6/q17 class).
   [claimed 20260830T182500Z]
   [done 20260830T182530Z -> 20260830T182530Z-grok-to-all-q1-batch-exact.md]

17. Claude ST 81 (to=all): fusion loses on RDNA3 via VGPR. Verify
    primary sources for the non-fusion dispatch cuts: llama.cpp PR 13014
    bs=1 MUL_MAT_ID, AMD-Ecosystem PR 59 24-VGPR MMVQ, and whether any
    published GDN/QSA prefill runs above batch 16 (Codex 162: Ember
    kQwen4ExpFrontierMoeMaxBatch=16 forces ~130 q16 chunks at 2074).
    Why: 345 tok/s is Amdahl-limited by that cap; tile A/B is parked.
    Send findings to=codex.
   [claimed 20260830T182600Z]
   [done 20260830T182640Z -> 20260830T182640Z-grok-to-codex-width-not-13014.md]

18. Claude ST 83: gap distribution is bimodal (p50 10.4 us, mean 52.3 us;
    top 1% = 41594 gaps = 61% of idle). Research published causes of
    hundreds-of-µs ggml HIP device stalls on RDNA3.5 (sync, pool growth,
    dependency waits) and how others found them. No lever from a mean.
    Send to=codex with falsifiers.
   [claimed 20260830T182800Z]
   [done 20260830T182820Z -> 20260830T182820Z-grok-to-codex-stall-causes.md]

19. Claude ST 85: 33675 copies (2.6%) each preceded by mean 3.1 ms stall
    = 48% of prefill idle. Which ggml HIP paths issue copyBuffer after
    host compute/sync? Check MoE id staging, sched splits, KV defrag.
    sync_fallback already 0 — do not re-blame it. Falsifiers + sizes.
    Send to=codex.
   [claimed 20260830T182930Z]
   [done 20260830T182950Z -> 20260830T182950Z-grok-to-codex-stalled-copies.md]

20. Codex 171 / Claude 86: keep intermediates on device. Does ggml HIP
    implement `set_tensor_async` / `get_tensor_async` without
    StreamSynchronize, or is async a lie that still blocks? Any
    published llama.cpp pattern that fuses subgraphs so host never
    sees the tensor? Why: 1.9x lever is the 35 set/get sites; wrong
    async API wastes a GPU run. Send to=codex.
   [claimed 20260830T183220Z]
   [done 20260830T183240Z -> 20260830T183240Z-grok-to-codex-async-set.md]

21. Claude 87 classifies gdn_eval_batch 6 as "pure staging". Verify
    against source: does the host consume qkv/output/recurrent after
    get? If yes, tranche 2 is not arithmetic-free. Send to=codex.
   [claimed 20260830T183400Z]
   [done 20260830T183410Z -> 20260830T183410Z-grok-to-codex-gdn-not-staging.md]

22. Claude 88 proposes tranche 0 async swap of gdn/moe/dense. Source
    already uses set_async/get_async + synchronize because host reads.
    Tell Codex not to re-do it. Why: wasted GPU A/B.
   [claimed 20260830T183520Z]
   [done 20260830T183520Z -> 20260830T183520Z-grok-to-codex-tranche0-done.md]

23. QSA tranche 1 needs `ggml_rope_multi` / `GGML_ROPE_TYPE_IMROPE` on
    HIP. Confirm the HIP backend actually implements IMROPE (not CPU
    fallback). Also: does ggml already have a device conv-window shift
    (`ggml_set`/`cpy`/`roll`) for GDN halo? Why: those two are the
    remaining 1.9x work; a missing HIP op would make VENDOR.md a trap.
    Send to=codex.
   [claimed 20260830T183610Z]
   [done 20260830T183630Z -> 20260830T183630Z-grok-to-codex-imrope-hip.md]

24. Map `ember_qwen_yarn_config` + host `rope()`/`rms_norm()` in
    qwen4exp_runtime to `ggml_rope_multi` arguments (sections, n_dims,
    freq_base, YaRN). Why: HIP IMROPE exists; a wrong sections vector
    silently mismatches host QSA and burns a GPU differential.
    Send to=codex.
   [claimed 20260830T183650Z]
   [done 20260830T183700Z -> 20260830T183700Z-grok-to-codex-yarn-imrope-map.md]

25. Codex 178: upstream precedent composing `ggml_rope_multi` with QSA
    projection/attention in one graph, IMROPE/YaRN preserved.
    [claimed 20260830T184900Z]
    [done 20260830T184900Z -> 20260830T184900Z-grok-to-codex-qsa-rope-graph.md]

26. PR27742 QSA selection is in-graph `ggml_top_k` + `ggml_set_rows`.
    Confirm HIP `supports_op` for both at QSA n_kv (often >1024).
    Why: copying llama.cpp selection without device TOP_K/SET_ROWS
    silently CPU-fallbacks and the host/device seam stays.
    Send to=codex.
    [claimed 20260830T185000Z]
    [done 20260830T185000Z -> 20260830T185000Z-grok-to-codex-topk-setrows-hip.md]

27. Map Ember host `int32_t positions[3]` / `position_history` to
    `ggml_rope_multi` src1 layout (`rope.cu` mrope/imrope indexing).
    llama.cpp uses I32 `[4 * n_tokens]`. Why: wrong pos tensor is a
    silent rotate, not a crash. Send to=codex.
    [claimed 20260830T185100Z]
    [done 20260830T185100Z -> 20260830T185100Z-grok-to-codex-mrope-pos-layout.md]

28. Confirm HIP `rope_multi` IMROPE pair layout matches
    `ember_qwen_yarn_apply` (NEOX halves of first 64, not sequential
    pairs). Why: a pair-stride mismatch is bit-wrong with no assert.
    Send to=codex.
    [claimed 20260830T185200Z]
    [done 20260830T185200Z -> 20260830T185200Z-grok-to-codex-imrope-pairs.md]

29. Claude 96: fill IMROPE lane 3 with physical token offset, not
    zeros. Verify against host indexer/QSA and `rope.cu` leftover
    axis. Why: 185 told Codex zeros; if wrong, correct now.
    Send to=codex.
    [claimed 20260830T185300Z]
    [done 20260830T185300Z -> 20260830T185300Z-grok-to-codex-lane3-zeros.md]

30. Claude 96 trap 3: pass `attn_factor=1.0` and let ggml derive it,
    vs 179 `config->attention_factor`. Check `ggml_rope_yarn` vs host
    `1+0.1*ln(factor)`. Why: silent scale on every rotated dim.
    Send to=codex.
    [claimed 20260830T185400Z]
    [done 20260830T185400Z -> 20260830T185400Z-grok-to-codex-attn-factor.md]

31. Map host `rms_norm` (`kEpsilon=1e-6`, mean-of-squares) to
    `ggml_rms_norm` eps and reduction axis for QSA heads (256) and
    indexer (128). Why: wrong eps or reducing over tokens is a
    silent numerics miss before RoPE. Send to=codex.
    [claimed 20260830T185500Z]
    [done 20260830T185500Z -> 20260830T185500Z-grok-to-codex-qsa-rms.md]

32. Verify `ggml_rope_multi` `c` is inverse-frequency or a
    `freq_factors` divisor. 179 Path 1 said `c=inv_freq`; if ggml
    divides theta by `c`, Path 1 as stated is silently wrong.
    Send to=codex.
    [claimed 20260830T185600Z]
    [done 20260830T185600Z -> 20260830T185600Z-grok-to-codex-c-not-invfreq.md]

33. Map host GDN conv-window shift (`conv` [3,10240]) to HIP
    `ggml_roll` / `ggml_set` so tranche 2 does not invent a kernel.
    Why: remaining 1.9x after QSA rope. Send to=codex.
    [claimed 20260830T185700Z]
    [done 20260830T185700Z -> 20260830T185700Z-grok-to-codex-gdn-conv-view.md]

34. Confirm HIP `supports_op` for `GGML_OP_SSM_CONV` and
    `GGML_OP_GATED_DELTA_NET` at GDN shapes. Why: existing GDN graph
    plus tranche 2 view/cpy are useless if conv/scan already CPU-fallback.
    Send to=codex.
    [claimed 20260830T185800Z]
    [done 20260830T185800Z -> 20260830T185800Z-grok-to-codex-gdn-ops-hip.md]

35. Confirm HIP `FLASH_ATTN_EXT` `supports_op` for Ember QSA
    attend shapes on gfx1151. Why: merging rope into the attend
    graph still CPU-fallbacks if FA is unsupported. Send to=codex.
    [claimed 20260830T185900Z]
    [done 20260830T185900Z -> 20260830T185900Z-grok-to-codex-qsa-fa-hip.md]

36. Confirm HIP `GET_ROWS` for F32 QSA K/V gather from cache using
    I32 top-k indices. Why: rope on device still uploads selected KV
    unless gather is a device op. Send to=codex.
    [claimed 20260830T190000Z]
    [done 20260830T190000Z -> 20260830T190000Z-grok-to-codex-get-rows-hip.md]

37. Confirm HIP `RELU` / `SUM_ROWS` / `MUL_MAT` for llama.cpp
    in-graph QSA indexer scoring (`build_qsa_top_k`). Why: TOP_K
    on HIP is useless if score reduce is CPU. Send to=codex.
    [claimed 20260830T190100Z]
    [done 20260830T190100Z -> 20260830T190100Z-grok-to-codex-indexer-ops-hip.md]

38. Compare `ggml_rope_yarn_corr_dims` to host
    `correction_low/high` (`qwen_yarn.c` truncate). Why: Path 2
    (`c=NULL`) is only safe if the ramps match. Send to=codex.
    [claimed 20260830T190200Z]
    [done 20260830T190200Z -> 20260830T190200Z-grok-to-codex-yarn-corr-dims.md]

39. Confirm `ROPE` `supports_op` (`nb[0]==type_size` and
    `contiguous_2`) still holds if Codex views the first 64 of a
    256-head instead of passing `n_dims=64` on the full tensor.
    Why: a view is the obvious way to spell partial rotary and
    would silently CPU-fallback. Send to=codex.
    [claimed 20260830T190300Z]
    [done 20260830T190300Z -> 20260830T190300Z-grok-to-codex-rope-view.md]

40. Can tranche 2 `ggml_cpy` next-conv into persistent
    `conv_history` (an INPUT) on HIP, or does gallocr/sched forbid
    overwriting inputs? Why: view mapping is useless if cpy dst
    cannot be the resident buffer. Send to=codex.
    [claimed 20260830T190400Z]
    [done 20260830T190400Z -> 20260830T190400Z-grok-to-codex-gdn-cpy.md]

41. Map `ggml_gated_delta_net` output layout to host
    `next_recurrent_state` so tranche 2 can keep it resident.
    Why: eval_batch still downloads gdn after attention_values
    (`frontier.cpp:1157-1160`). Send to=codex.
    [claimed 20260830T190500Z]
    [done 20260830T190500Z -> 20260830T190500Z-grok-to-codex-gdn-state.md]

42. Claude 213 CPU oracle 1e-7. HIP `rope_theta_fp64` vs CPU
    `float theta *= theta_scale`. Why: oracle does not prove HIP
    at long pos / 1e7 base. Send to=codex.
    [claimed 20260830T190600Z]
    [done 20260830T190600Z -> 20260830T190600Z-grok-to-codex-rope-f32-f64.md]

43. Does HIP `gated_delta_net` inplace write `src_state` only after
    the last token, or does it clobber `s_d` mid-scan? Why: 213
    recommended inplace; a WAR bug would scramble every GDN layer.
    Send to=codex.
    [claimed 20260830T190700Z]
    [done 20260830T190700Z -> 20260830T190700Z-grok-to-codex-gdn-inplace-war.md]

44. After GDN inplace, what does `eval_batch` still download besides
    logits? Why: tranche 2 cannot drop copies it has not named.
    Send to=codex.
    [claimed 20260830T190800Z]
    [done 20260830T190800Z -> 20260830T190800Z-grok-to-codex-gdn-remaining-gets.md]

45. Name the host ops between QSA project / rotate / attend that
    force the five+two gets. Why: 219 said QSA is the copy wall;
    fusion needs the seams named. Send to=codex.
    [claimed 20260830T190900Z]
    [done 20260830T190900Z -> 20260830T190900Z-grok-to-codex-qsa-host-seams.md]

46. Claude 217: HIP-vs-exact at pos 262141 should be ~1e-6
    (cosf after fp64 mod-2pi), not the CPU graph's 1.86e-3.
    Why: if HIP stays at 1e-3, it did not take rope_theta_fp64.
    Send to=codex.
    [claimed 20260830T191000Z]
    [done 20260830T191000Z -> 20260830T191000Z-grok-to-codex-hip-rope-floor.md]

47. Claude 219 census counts `qsa_rotate_q1` as 2 barriers.
    Rotation ctx is only built if `self_k_rot`/`self_v_rot`
    exist. Why: F32 path may already skip those 2. Send to=codex.
    [claimed 20260830T191100Z]
    [done 20260830T191100Z -> 20260830T191100Z-grok-to-codex-rotate-optional.md]

48. Does the published Qwen3.8-Flash GGUF contain
    `attn_k_rot.weight` / `attn_v_rot.weight`? Why: 225 said
    rotate barriers may be dead; need a checkable yes/no.
    Send to=codex.
    [claimed 20260830T191200Z]
    [done 20260830T191200Z -> 20260830T191200Z-grok-to-codex-no-k-rot.md]

49. Map the QSA projection graph nodes vs `prepare_qsa_row`
    so Codex can insert split/RMS/yarn without a host get.
    Why: 227 said that 5-get is the cheapest cut. Send to=codex.
    [claimed 20260830T191300Z]
    [done 20260830T191300Z -> 20260830T191300Z-grok-to-codex-proj-insert.md]

50. Claude 227 says moving RMS+yarn into projection deletes
    the :1513 get entirely. finish_qsa_row still reads iq/ik.
    Why: barrier count vs get count. Send to=codex.
    [claimed 20260830T191400Z]
    [done 20260830T191400Z -> 20260830T191400Z-grok-to-codex-five-get-not-zero.md]

51. Propose dead-code register entry for HIP graph replay.
    Why: user rule; LOOP already forbids re-opening it but it
    is not in docs/dead-code-candidates.md. Send to=claude.
    [claimed 20260830T191500Z]
    [done 20260830T191500Z -> 20260830T191500Z-grok-to-claude-graphs-register.md]

52. ggml op recipe for `finish_qsa_row` 4-token pool / ReLU
    score / top-512, so iq/ik can stay on device. Why: 231
    said tranche 1 cannot drop the :1513 barrier without it.
    Send to=codex.
    [claimed 20260830T191600Z]
    [done 20260830T191600Z -> 20260830T191600Z-grok-to-codex-dense-2048.md]

53. Can HIP `SET_ROWS` write current K/V/`index_key` into a
    persistent INPUT cache, like GDN `cpy` into `conv_history`?
    Why: 235 said that kills :1513 at ctx<=2048. Send to=codex.
    [claimed 20260830T191700Z]
    [done 20260830T191700Z -> 20260830T191700Z-grok-to-codex-set-rows-input.md]

54. Does HIP `flash_attn_ext` require the host pad to
    `qsa_cached_width`, or can dense attend pass exact
    `n_tokens`? Why: pad+upload is the remaining attend
    barrier after SET_ROWS. Send to=codex.
    [claimed 20260830T191800Z]
    [done 20260830T191800Z -> 20260830T191800Z-grok-to-codex-fa-pad.md]

55. Claude 231 RMS oracle + still-12 barriers. Point at 235:
    ctx<=2048 iq get is unused, so 5->1 not 5->2. Send to=claude.
    [claimed 20260830T191900Z]
    [done 20260830T191900Z -> 20260830T191900Z-grok-to-claude-rms-ack.md]

56. With rotate dead and scorer dead at ctx<=2048, can
    project+SET_ROWS+attend be one ggml graph? Why: that
    deletes the remaining QSA barrier pairs. Send to=codex.
    [claimed 20260830T192000Z]
    [done 20260830T192000Z -> 20260830T192000Z-grok-to-codex-qsa-fuse.md]

57. After GDN inplace, must `graph->output` still download
    or can MoE consume it on-device? Why: that is the last
    GDN get once qkv/recurrent stay resident. Send to=codex.
    [claimed 20260830T192100Z]
    [done 20260830T192100Z -> 20260830T192100Z-grok-to-codex-gdn-out-hc.md]

58. Claude 233: below 2048 only `state.index_key.size()` is
    consumed. SET_ROWS needs a token counter, not a host
    vector. Send to=codex.
    [claimed 20260830T192200Z]
    [done 20260830T192200Z -> 20260830T192200Z-grok-to-codex-ik-counter.md]

59. ggml recipe for host `hc_combine` (4-stream mix) so GDN
    output need not download. Why: 245 named it as the last
    GDN get. Send to=codex.
    [claimed 20260830T192300Z]
    [done 20260830T192300Z -> 20260830T192300Z-grok-to-codex-hc-combine.md]

60. Claude 235: Qwen MMVQ ceiling. Physical width for
    logical 3 is 5 on dense/moe. Send to=codex.
    [claimed 20260830T192400Z]
    [done 20260830T192400Z -> 20260830T192400Z-grok-to-codex-physical-q.md]

61. How is green-at-2 launched — serial q=1 or
    dense_cached_width(2)=5? Why: 251 said 2 and 3 share
    physical 5; that only matters if 2 uses that cache.
    Send to=codex.
    [claimed 20260830T192500Z]
    [done 20260830T192500Z -> 20260830T192500Z-grok-to-codex-q3-not-ceiling.md]

62. Second cause for q3 vs q2: sequence-length path, not
    MMVQ. GDN conv or QSA at n_tokens=3. Send to=codex.
    [claimed 20260830T192600Z]
    [done 20260830T192600Z -> 20260830T192600Z-grok-to-codex-q3-seq.md]

63. Claude 237: at which widths was ncols=5 bit-exact?
    Codex 106 is the table. Then whether MoE pad can
    leak (239 NaN fill). Send to=claude.
    [claimed 20260830T192700Z]
    [done 20260830T192700Z -> 20260830T192700Z-grok-to-claude-ncols5-widths.md]

64. Claude 241: dense pad clean. Can moe_batch run CPU
    F32? GDN@3 vs serial source. Send to=claude.
    [claimed 20260830T192800Z]
    [done 20260830T192800Z -> 20260830T192800Z-grok-to-claude-moe-cpu-gdn.md]

65. Claude 243 taking GDN@3: already green. Next
    stateful: PLE dilation or rope/pos at n=3.
    Send to=claude.
    [claimed 20260830T192900Z]
    [done 20260830T192900Z -> 20260830T192900Z-grok-to-claude-gdn-already.md]

66. Claude 245: FORCE_Q1 at width 3. Width 2 is NOT
    q=1 graphs. Correct the fail branch. Send to=codex.
    [claimed 20260830T193000Z]
    [done 20260830T193000Z -> 20260830T193000Z-grok-to-codex-force-q1.md]

67. Does run_qsa/run_gdn use cur_pos or the full
    pre-pushed mrope_positions during batch_layer_q1?
    Why: mask-31 fail branch. Send to=claude.
    [claimed 20260830T193100Z]
    [done 20260830T193100Z -> 20260830T193100Z-grok-to-claude-qsa-curpos.md]

68. Claude 249 taking PLE/hc_rows/QSA append. Leftover
    from 265: hc_combine at n=3. Send to=claude.
    [claimed 20260830T193200Z]
    [done 20260830T193200Z -> 20260830T193200Z-grok-to-claude-hc-combine.md]

69. Claude 251 taking hc_mix/run_moe. 267 already
    closed hc_combine. Confirm run_moe has no state.
    Send to=claude.
    [claimed 20260830T193300Z]
    [done 20260830T193300Z -> 20260830T193300Z-grok-to-claude-moe-stateless.md]

70. If FORCE_Q1 width 3 passes: bisect order. MoE/HC
    already CPU-green; Attention=4 first. Send to=codex.
    [claimed 20260830T193400Z]
    [done 20260830T193400Z -> 20260830T193400Z-grok-to-codex-bisect-order.md]

71. Claude 253 green branch too narrow: GDN HIP S_v=128
    is not dense_eval_rows. MASK=4 still first.
    Send to=codex.
    [claimed 20260830T193500Z]
    [done 20260830T193500Z -> 20260830T193500Z-grok-to-codex-green-not-dense.md]

72. Red-branch leftovers: prefill_chunk_rows and embed
    at n=3. Why: 273 named them if mask 31 is red.
    Send to=claude.
    [claimed 20260830T193600Z]
    [done 20260830T193600Z -> 20260830T193600Z-grok-to-claude-chunk-embed.md]

73. Claude 255: reload LOOP.md after cc2d3d9. ACK.
    [claimed 20260830T193700Z]
    [done 20260830T193700Z -> 20260830T193700Z-grok-to-claude-loop-reload.md]

74. HIP gated_delta_net S_v=128: n=3 vs n=1 source
    diffs. Why: uncovered if MASK=4 is green. Send
    to=codex.
    [claimed 20260830T193800Z]
    [done 20260830T193800Z -> 20260830T193800Z-grok-to-codex-gdn-hip.md]

75. Type-101 MMVQ: does ncols=3 take a different
    kernel than ncols=2? Why: QSA exact n, 2 green
    3 red. Send to=codex.
    [claimed 20260830T193900Z]
    [done 20260830T193900Z -> 20260830T193900Z-grok-to-codex-mmvq-ncols3.md]


































































































































