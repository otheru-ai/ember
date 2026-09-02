# rocm-systems runtime review: what the HIP/ROCr dispatch path offers gfx1151

Read 2026-09-02 against `ROCm/rocm-systems` branch `develop`, commit
`1b648038a0ac164cf2f06f2a581ced12cf5f7378`. Paths below are relative to
`projects/` in that tree. Ember runs on gfx1151 (Strix Halo, RDNA 3.5, one
WGP pool, unified memory) with the `ember-rocm:10.0` image; the shipped
`libamdhip64.so.7` / `libhsa-runtime64.so.1` were grepped for every env
name cited so that nothing below proposes a knob the binary cannot read.

Headline: **nothing in the runtime is a structural lever for this part.** The
new dispatch-path work in CLR is gated to gfx94x / gfx12.5, our defaults are
already the fast ones, and the one remaining host-side cost — the
spec-cycle boundary after argmax readback — is a program-structure cost,
not a runtime setting. Two cheap env A/Bs remain, both host-only and
harmless; a third "fast copy" knob is a simulator feature and must not be
used on hardware.

## Dead on gfx1151 — do not spend time here

| feature | gate | where |
|---|---|---|
| KDQ kernel-descriptor prefetch (`DEBUG_CLR_ENABLE_KDQ`) | gfx12.5 only | `rocr-runtime/.../core/runtime/amd_aql_queue.cpp:119-126`; queue side `clr/rocclr/device/rocm/rocdevice.cpp:3500-3625` |
| extended dispatch packet | gfx12.5 only | `clr/rocclr/device/rocm/rocsettings.cpp:188` |
| device-memory kernargs / device ring buffer | gfx94x path; gfx11 stays on `HostKernelArgs` | `rocsettings.cpp:255-285` |
| doorbell skip (`DEBUG_CLR_DOORBELL_SKIP`) | forced off under `IS_LINUX` | `clr/rocclr/device/rocm/rocvirtual.cpp:1595-1598` |

`DEBUG_CLR_ENABLE_KDQ` and `DEBUG_CLR_AQL_BARRIER_OPT` are also absent from
the 10.0 image binaries (`grep -c -a -o` on the `.so.7` / `.so.1`), so even
the string is not there to set.

## Defaults already optimal — verified, not assumed

* **AQL packet write uses MOVDIR64B** when the CPU reports it
  (`rocdevice.cpp:1695`; `DEBUG_CLR_USE_MOVDIR64B=1` default,
  `clr/rocclr/utils/flags.hpp`). albion/otheru's CPU has `movdir64b` in
  `/proc/cpuinfo`. One 64-byte non-temporal store per packet is the floor.
* **Active (spin) wait is the HIP default** (`clr/hipamd/src/hip_context.cpp:52`
  sets `ActiveWait(true)`; `clr/rocclr/device/device.hpp:2409`). With active
  wait, completion signals are GPU-only `BusyWaitSignal`s
  (`rocvirtual.cpp:701-717`) and `WaitForSignal` (`rocvirtual.hpp:55-95`)
  spins rather than taking the interrupt path. So `HSA_ENABLE_INTERRUPT` is
  already irrelevant to our sync points.
* **`AMD_OPT_FLUSH=1`** makes gfx11 release fences agent-scope rather than
  system-scope (`rocvirtual.cpp:2423-2440`). Default on.
* **Pageable H2D copies** go through `KernelBlitManager::writeBuffer`
  (`clr/rocclr/device/rocm/rocblit.cpp:2257-2320`): staging memcpy plus the
  `__amd_rocclr_copyBuffer` blit kernel. Blocking variants add a
  system-scope fence (`addSystemScope()`). This is already the cheapest
  path for a non-pinned source; pinning the source is the program-side fix
  and Ember's request path already stages into pinned buffers.

## Trace check: blit copies are not the idle

From the retained DS4 decode trace
(`otheru:/root/ember/reports/profile-live/trace-decode_kernel_trace.csv`,
Aug 22, 256-token count-up probe): `__amd_rocclr_copyBuffer` is ~10% of
dispatches and its inter-dispatch p50 gap is 1.8 µs — the same as
kernel→kernel. The idle sits at the 67 spec-cycle boundaries: p50 2.3 ms,
p90 32 ms, each following the argmax → three synchronous readbacks. That is
the previously filed host-boundary finding, reconfirmed from the
distribution, and it is not a runtime knob.

## Two env A/Bs worth one run each (codex runs GPU)

Both names are present in the 10.0 image libraries. Both are host-only.
Read them from the per-stream gap distribution (p50/p90/p99 of dispatch
gaps at spec-cycle boundaries), not from total tok/s — the expected effect
is microseconds per sync, which a tok/s mean will not resolve.

1. **`DEBUG_CLR_DIRECT_DOORBELL=1`** (default 0; `rocdevice.cpp:3598`,
   `clr/rocclr/utils/nontemporal.hpp:217-243`). Writes the doorbell
   directly instead of through `hsa_signal_store`. Falsifier: boundary-gap
   p50/p90 unchanged within run-to-run spread → drop it.
   **Result 2026-09-02 (codex, `/srv/models/perf/rocm-runtime-env-ab-44e7f5e-20260902-r5-kernel-env-arms/doorbell`):
   the 32-token warm-up completed with speculation, then the single
   256-token measured request produced zero bytes for 900 s (curl exit 28).
   No gap statistic exists. Not a perf number and not a "no gain": the
   direct-doorbell path stalls the spec loop on this stack. Dropped and
   moved to "Do not use"; not rerun.**
2. **`HSA_ENABLE_MWAITX=0`** (default on, `rocr-runtime/.../core/util/flag.h:295`;
   `g_use_mwaitx = flag_.check_mwaitx(cpuinfo.mwaitx)`,
   `core/runtime/runtime.cpp:2676`). `BusyWaitSignal::WaitRelaxed`
   (`core/runtime/default_signal.cpp:80-112`) calls
   `timer::DoMwaitx(&signal_.value, value, 60000, true)` on every
   iteration regardless of the wait hint (the interrupt path uses 1000 for
   ACTIVE, `interrupt_signal.cpp:150-203`). `DoMwaitx` is
   `_mm_monitorx` + `_mm_mwaitx(MWAITX_ECX_TIMER_ENABLE, 0, timeout)`
   (`core/inc/signal.h:114-123`); the EBX timeout is in TSC ticks, so 60000
   is ~16 µs at otheru's ~3.8 GHz TSC. The monitor wakes on the GPU's
   store, so the timeout is a bound on a missed wake, not an added latency
   per wait — the expected effect is small or nil. Falsifier: same as
   above. Worth one run only because the wait sits on every spec-cycle
   boundary and the run is free.

## Do not use

**`DEBUG_CLR_DIRECT_DOORBELL=1`** — hung the measured request after a
working warm-up (above). One occurrence, unexplained, on ROCm 10 dev image
`44e7f5e`; the cost of a second look is a 15-minute GPU hang, so it is
closed rather than diagnosed.

**`HSA_ENABLE_DTIF_FAST_COPY`** exists in the 10.0 binary but is a DTIF
simulator-platform feature (`core/inc/thunk_loader.h:54`,
`flag.h:343-355`). Its memcpy substitute has no GPU ordering
(`core/runtime/amd_blit_kernel.cpp:617-667`). On hardware it would silently
race copies against kernels.

## Program-side item found while reading

`engine/ggml/src/ggml-cuda/fattn-sparse.cu` allocates six device buffers
with `cudaMallocAsync` and frees them per call. That would be a real
per-dispatch cost — but the op has no producer in the tree. Filed as
dead-code entry 8 in `docs/dead-code-candidates.md`, not as a perf finding.
