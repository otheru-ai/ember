# ROCm 10.0 toolchain and roofline migration

Ember's development and release builds use the digest-pinned
`rocm/dev-ubuntu-24.04:10.0.0-full` toolchain. The dev image asserts that it
contains both `rocprofv3` from ROCm 10.0.0 and ROCm Compute Profiler 3.8.0;
the stripped release image still contains only the server's recursive runtime
dependency closure.

Every real `scripts/profile_gpu.sh` run archives the resolved local image ID
and any registry repo digests, the full version output from `rocprofv3` and
`rocprof-compute`, and the device-specific output of
`rocprofv3-avail info --pmc -d 0`. `manifest.json` embeds the selected
`FETCH_SIZE`/`WRITE_SIZE` definitions and hashes the complete counter-info
file. A locally built `ember-rocm:10.0-dev` commonly has no repo digest, so the
content-addressed image ID remains mandatory provenance rather than treating an
empty `RepoDigests` list as an error.

This upgrade matters on Ember's gfx1151 target because ROCm Compute Profiler
3.8.0 corrects the roofline precision set and APU machine specification for
gfx1150, gfx1151, and gfx1152. gfx1153 roofline collection is not supported by
that release. These are profiler-correctness improvements, not evidence of an
Ember throughput improvement.

## Profiling contract

Keep `scripts/profile_gpu.sh` as the authoritative Ember decode-bandwidth
harness. It deliberately collects timing and performance counters in separate
passes, isolates phases with an idle gap, and reports the raw segments.

For legacy bundles, `profile_report.py` interprets `FETCH_SIZE` and
`WRITE_SIZE` as KiB because that unit was measured on gfx1151 under ROCm 7.14.
AMD's ROCm 10 documentation defines `Counter_Value` as numeric but does not
state the unit for these two derived counters. Consequently, ROCm 10 bundles
carry `counter_unit.status: uncertified_rocm10_gfx1151` and
`release_bandwidth_verdict_certified: false`. Their raw profiles remain useful,
but the analyzer withholds its roofline verdict until a known-traffic gfx1151
microbenchmark confirms both counters and the operator passes its JSON with
`--counter-calibration`. Preserve the raw `rocprofv3-counter-info.txt` in that
calibration record.

The reproducible calibration workload is `tools/bench_counter_traffic.hip`.
Run its collector on the exclusively held target:

```bash
scripts/calibrate_counter_units.sh --image ember-rocm:10.0-dev \
  --out-dir /var/tmp/ember-counter-calibration-<stamp>
```

The collector profiles `FETCH_SIZE` against read-only 64-byte cache-line
streams and `WRITE_SIZE` against write-only streams, with a no-memory baseline,
three buffer sizes larger than MALL, and one PMC pass per counter. It writes
raw CSVs, `samples.jsonl`, and a regression result. Run 33288846711 on gfx1151
with ROCm 10.0 certified `FETCH_SIZE` as 64-byte transactions (65.137 fitted,
1.78% error) and `WRITE_SIZE` as 128-byte transactions (129.490 fitted, 1.16%
error); both regression residuals were below 0.13%. The bounded record used by
the analyzer is `share/benchmark/gfx1151-rocm10-counter-calibration.json`.
Do not replace those distinct scales with one shared unit, or infer a unit from
the model's nominal weight size or a PMC-pass duration; those are different
quantities.

ROCm Compute Profiler's roofline is a complementary machine-ceiling
measurement:

```bash
rocprof-compute profile --name ember-gfx1151-roofline --roof-only -- \
  /ember/build-rocm/ember-dflash --help
```

Use `--bench-only` when only the roofline microbenchmark is wanted. Analyze the
result with ROCm Compute Profiler and select only a precision that 3.8.0 reports
as supported for the device; the corrected tool no longer offers unmeasurable
precisions for gfx1151.

Do not merge a ROCm 7.x workload directory into a ROCm 10.0 report. The 3.8.0
system-information schema is incompatible with older saved workloads, and the
tool now requires `--overwrite` when profiling into a non-empty output path.
Create a fresh directory and retain its raw files in the release benchmark
bundle.

## Release gate

Building the container proves toolchain compatibility only. Before publishing
a ROCm 10-based image, rerun the differential validator and the full profiling
and performance bundle on an exclusively held gfx1151 host with the release
model. Keep all pre-existing ROCm 7.14 measurements labelled as historical;
do not relabel them or infer a speedup from the toolchain version.

Primary references:

- [ROCm 10.0.0 release notes](https://rocm.docs.amd.com/en/docs-10.0.0/about/release-notes.html)
- [ROCm Compute Profiler profiling](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.0.0/how-to/profile/mode.html)
- [ROCprofiler-SDK CLI options](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-10.0.0/quick-reference/rocprofv3-cli-options.html)
- [`rocprofv3-avail` counter metadata](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-10.0.0/how-to/using-rocprofv3-avail.html)
