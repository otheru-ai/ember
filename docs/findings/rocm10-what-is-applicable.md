# ROCm 10: what is actually new for us

Read 2026-09-02 against the ROCm 10.0 changelog and the RDNA3.5 system
optimization guide.

## Already applied — no gain available

The RDNA3.5 optimization guide makes three recommendations for gfx1151. Checked
on otheru; we already meet all three, so nobody should spend time here:

| recommendation | ours |
|---|---|
| keep BIOS VRAM reservation small (0.5 GB) | **536870912 B = exactly 0.5 GiB** |
| raise GTT instead, via `amd-ttm` / `ttm.pages_limit` | **32505856 pages = 124.0 GiB** of 125 |
| kernel >= 6.18.4 (or 6.17 HWE) for gfx1151 stability | **7.1.3** |

The guide's point is that RDNA3.5 shares memory physically, so GTT-backed
allocation beats VRAM reservation. That is the configuration we run.

## Genuinely new and worth having

**PC sampling on gfx115x now reports a Dual VALU (VOPD) instruction-mix metric
in the WGP panel.** This is the one item that changes what we can know. Our
kernel work concluded VOPD is unreachable from inline asm and that DPP is where
the win is — reasoning from the ISA, with no way to measure dual-issue occupancy
on a real kernel. This metric turns that into a measurement, and it can be
pointed at the hand-written assembly we already ship.

Not verified in our image: the `ember-rocm:10.0` images are not in the podman
store I can read, so whether `rocprofv3 --list-avail` exposes the counter here
is unconfirmed. That check belongs to whoever owns the images.

## Possibly useful, unproven

* **Green contexts** — `hipGreenCtxCreate`, `hipDevSmResourceSplit` partition
  compute resources. The obvious application is running drafter and target
  concurrently rather than serially, which is where speculative decode spends
  its time. Speculative; no evidence it helps on an APU with one WGP pool.
* **`hipEventDisableTiming`** — cheaper event records where the timestamp is
  unused. Free if we record events we never read.
* **rocBLAS per-batch alpha/beta** on level-2 batched ops. We are custom-kernel
  and ggml, so probably irrelevant.

## Does NOT reopen a closed decision

The changelog improves **graph replay for asynchronous allocations** —
allocation nodes no longer block during replay. That is not why HIP graph replay
was ruled out here: the blocker was a measured **1.84 us/node floor on gfx1151**
plus an observed regression, which an allocation-node fix does not touch. See
the dead-code register kept on the branch that owns that runtime (entry 4).
Do not re-open it on this changelog line.

## Breaking change to watch

AMD SMI bumped to `libamd_smi.so.27`. Anything linked against `.so.26` must be
relinked — worth checking before the container cutover, not after.
