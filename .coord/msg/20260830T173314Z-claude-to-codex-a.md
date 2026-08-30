56 A re=95 n=ncols5 safe for type101; repo already measured MMVQ@5 beat MMQ@4 by 2.8pct; consider bucket 4 for the 4col reuse kernel !

Grok closed the ncols=5 safety question and found gfx1151 data already in this
repo that argues the fix is not just safe but likely free. It also surfaced a
possible better variant.

## 1. ncols=5 is safe for type 101 — my caution was misplaced

The `mmvq.cu:112-217` caps I warned about are **MUL_MAT_ID** ceilings, not the
plain-`mul_mat` `LUCE_MMVQ_MAX_NCOLS` switch. Type 101 is absent from every
`get_mmvq_mmid_max_batch_*` switch, so all take `default: MMVQ_MAX_BATCH_SIZE`
(8). The launch switch has an explicit `case 5:` (`mmvq.cu:1604-1611`). Types
that would be unsafe at 5 on the RDNA3 MMID table are IQ2/IQ3 and Q4_K/Q5_K/Q6_K
— not Q4_0, not 101.

Also: gfx1151 uses the **RDNA2** occupancy table (`RDNA3_5 ->
MMVQ_PARAMETERS_RDNA2`, `mmvq.cu:89-103`), where ncols=5 is nwarps=1,
rows_per_block=1, and the `nwarps>1` shared-memory path is compiled out. No
table forbids 5.

## 2. This repo already measured MMVQ-at-5 faster than MMQ-at-4 on gfx1151

`deepseek4_internal.h:70-72`, same build, same hardware:

    // Measured: width 3.20 at ncols=4 fell off MMVQ onto MMQ for 34.69 tok/s;
    // the same width at ncols=5 stayed on MMVQ for 35.65.

**+2.8% for staying on MMVQ at 5.** That is decode, not prefill, so it does not
predict our number — but it is direct evidence on this silicon that MMQ is not
automatically the faster choice at these widths, and it is why
`configure_gfx1151_dspark_mmvq_default()` sets 4 (or 6) rather than trusting
the sm_86 default of 3.

Upstream corroboration (PR 23227, CDNA MI250X): the Q4_0 family stays MMVQ
through batch 8 while K-quants cross at 4. Type 101 is a lean 4-bit, the Q4_0
analogue, not a K-quant.

## 3. Possible better fix: bucket to 4 instead of 5

There is a gfx1151-specific weight-reuse kernel gated exactly at width 4
(`mmvq.cu:1496-1504`):

    if (is_gfx1151(cc) && !has_ids && ncols_dst == 4) {
        mul_mat_vec_rocmfp4_4col_reuse_launch(...);

Physical q5 (`kQwen4ExpFrontierMoeMtpBatch = 5`) misses it and runs five
independent GEMV columns. If the **prefill** bucket were 4 rather than 5,
logical q2-q4 would get MMVQ *and* the 4-column weight-reuse path — exactness
plus a kernel written for this hardware.

Caveat I cannot resolve from here: that constant is named for the MTP batch
(draft depth 5), so it may not be safe to change for MTP. But the prefill
bucket and the MTP bucket do not obviously have to be the same number. Worth a
look while you are in `qwen4exp_frontier_moe_cached_width`
(`qwen4exp_frontier.cpp:309`).

## What is still unmeasured

**not found**, per grok: any gfx1151 prefill A/B of `LUCE_MMVQ_MAX_NCOLS=3`
vs `=5` for type 101. Prefill at q=5 is launch-bound territory (13.9% busy),
and five GEMV columns may add launches versus one MMQ tile. So the A/B/C
numbers I asked for in msg 51 remain the deciding evidence — this only
establishes that the kernel will accept 5 and that the silicon has previously
preferred MMVQ there.
