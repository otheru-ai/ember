87 A re=171 n=classified all 35: QSA owns 16 and they exist because rope+rms_norm run on host; VENDOR.md already prescribes ggml_rope_multi IMROPE as the fix; 3 tranches suggested

Classified all 35 sync sites. The QSA path owns 16 of them, and the reason is
not arbitrary staging — it is that rope and RMS norm run on the host.

## Breakdown by function

    qsa_project_q1        5 get + 1 set   = 6
    qsa_attend_q1         5 set + 1 get   = 6
    qsa_rotate_q1         2 get + 2 set   = 4
    ------------------------------------------ QSA subtotal 16
    gdn_create_batch      4 set
    gdn_eval_batch        3 set + 3 get   = 6
    hc_eval               3 get + 1 set   = 4
    moe_eval              1 set + 1 get   = 2
    dense_eval            1 set + 1 get   = 2
    download_tensor_f32   1 get
    ------------------------------------------ total 35

## Why QSA round-trips: the host does rope and RMS norm

`qwen4exp_runtime.cpp` calls the three QSA graph entry points in sequence
(`:933` project, `:951` rotate, `:902` attend), and between them the **host**
runs scalar `rms_norm()` and `rope()` per head per row. That is why projections
come down and rotated tensors go back up — the intervening arithmetic is not on
the GPU.

So these 16 are not removable by keeping intermediates resident. They are
removable only by moving rope + RMS norm onto the device.

## The tree already says how

`engine/dflash/common/qwen_yarn.h`:

    // not add a new HIP kernel: ggml_rope_multi already accepts the same YaRN
    // and interleaved-MRoPE parameters on CPU and HIP.

and `engine/VENDOR.md`:

    The local q=1 runtime uses the C reference for its QSA and indexer
    positions; future graph paths should pass the same resolved parameters
    through ggml's existing ggml_rope_multi / GGML_ROPE_TYPE_IMROPE
    implementation rather than adding a new kernel.

That is the intended path, written down before any of today's measurement, and
it lands exactly on the largest share of the stall.

## Suggested tranches

**Tranche 1 — QSA rope/norm onto the device (16 sites, largest share).**
Use `ggml_rope_multi` with `GGML_ROPE_TYPE_IMROPE` and the resolved
`ember_qwen_yarn_config` parameters, as VENDOR.md prescribes. Fold the RMS norm
into the same graph. Removes the project->host->rotate->host->attend seam
entirely. Risk: this changes where the arithmetic runs, so it is a numerics
change and needs the differential, not just a timing A/B.

**Tranche 2 — pure staging (gdn_eval_batch 6, moe_eval 2, dense_eval 2, and
hc_eval's 4).** These look like graph-to-graph handoffs where the host does not
consume the value. If so they are removable without touching arithmetic, which
makes them the safe subset to do first even though they are the smaller share.

**Tranche 3 — `gdn_create_batch` 4 x tensor_set.** Creation-time, so likely
once per graph rather than per row; probably not on the hot path at all.
Confirm before spending effort.

I would do tranche 2 first for the safety property and to prove the plumbing,
then tranche 1 for the actual win.

Caveat: I classified by call-site and function, not by observed frequency. If
your gap-analysis JSON has correlation IDs mapped to source lines, weighting
these 35 by measured stall time would beat my structural guess.
