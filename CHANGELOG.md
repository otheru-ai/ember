# Changelog

Ember uses calendar versions in `YEAR.MONTH.DAY` form without zero-padding.
Release notes record user-visible features, fixes, compatibility changes,
upgrade steps, and validated hardware. Git tags add a `v` prefix; for example,
version `2026.8.10` is tagged `v2026.8.10`.

`VERSION` is authoritative. Ember publishes at most one release per calendar
day; additional fixes remain on `main` until the next dated release rather than
using an ambiguous same-day suffix.

## Unreleased

## 2026.8.24

### Added

- **bench:** automate the per-release performance bundle (`ac741304`)
- **ci(gfx1151):** benchmark the certified image as part of certification (`adf64ec4`)
- **perf:** a browsable performance page, built from benchmark bundles (`f9dfa52a`)

### Changed

- **deepseek4:** qualify the wide verify path on a rate, not a lucky streak (`b66595d3`)
- **ggml:** stop launching one block per row in `reduce_rows_f32` (`00556dd2`)
- **ggml:** size the rope block to the row instead of a fixed 256 (`57f11cd8`)
- **ggml:** collapse the short-row reduction to its exact minimal form (`f3da20c9`)
- **bench:** re-measure 2026.8.10, .22 and .23 against the current model (`f8805587`)

### Fixed

- **deepseek4:** hand back to AR when batch-verify warmup is unreachable (`b2428026`)
- **deepseek4:** abandon speculation after one unqualified profitability pause (`d616074f`)
- **deepseek4:** bound the warmup exclusion so the profit scheduler can still see (`d8993ed3`)
- **bench:** plot the depth that was asked for, not the one that drifted (`dba6c38f`)
- **bench:** benchmark an old release as it shipped, and keep failure reasons (`212bf2af`)
- **bench:** repair four defects the first dry run exposed (`6c976290`)

## 2026.8.23

### Fixed

- **test:** assert the compose environment syntax the file actually uses (`5f6241cc`)
- **deepseek4:** stop charging warmup cost to steady-state speculation (`af16fe4b`)
- **docker:** drop compose declarations that only restated engine defaults (`7810971a`)
- **docker:** stop the entrypoint overriding compose, make compose authoritative (`8c999fee`)
- **engine:** correct two profiling-harness bugs found on hardware (`3da625c0`)

### Performance

- **deepseek4:** 33.60 -> 37.49 tok/s decode via width-6 verify (`027ed106`)
- **engine:** extend DPP wave reductions, verified against the RDNA 3.5 ISA (`3a54084c`)
- **engine:** DPP wave reduction in hand-written assembly for gfx1151 (`5f460a0c`)
- **engine:** select WGP vs CU mode per translation unit on gfx1151 (`410d61d1`)
- **engine:** branch-free UE4M3 scale decode for gfx1151 (`b6291ce6`)

### Build and CI

- **docker:** add Hyperloom to the dev image (`88898ac8`)

### Documentation

- **perf:** document TTFT, and record why narrowing the FA KV read failed (`1cb30f76`)
- **engine:** link the two competing LUCE_MMVQ_MAX_NCOLS defaults (`4a1ccfef`)
- **engine:** correct the HIP-graph diagnosis, which blamed the wrong tensor (`dcc00326`)

### Maintenance

- **release:** add gfx1151 kernel profiling harness (`e5a0d064`)
- **release:** document CI gates and clear prune residue (`f02f7d10`)
- **release:** document Forgejo sanitizer split (`b760adb8`)
- **release:** stabilize Forgejo queue (`02305c07`)
- **release:** recover release-note publication (`08f4f9f8`)
- **release:** repair certified tag publication (`45ac7d3c`)
## 2026.8.22

### Curated notes

- Added `--host` and `EMBER_HOST` so trusted container/Kubernetes gateways can
  reach Ember while preserving the unauthenticated loopback default.
- Added `EMBER_VERIFY_EXISTING_SHA256=0` for trusted immutable model stores;
  downloaded artifacts remain pinned and checksum-verified before promotion.
- Integrated and documented the opt-in CPU/GPU/XDNA2 resident DSpark
  prototype. The XDNA Compose overlay now selects the measured two-session
  proposal pipeline instead of the slower target-expert placement experiment.
- Kept heterogeneous inference experimental: the fixed throughput fixture
  passed, while capture-graph output equivalence remains a promotion blocker.
- Linked README installation guidance to the current GitHub release instead of
  embedding a version that becomes stale after every release.
- Made candidate certification and release promotion fully unattended: trusted
  `main` commits proceed from immutable GHCR images through the dedicated
  gfx1151 runner to changelog generation, tagging, and package publication.
- Pinned the certification model pair to the published Hugging Face artifacts,
  added IOMMU/device preflight checks, and made the hardware job quiesce and
  restore production around exclusive GPU validation.
- Upgraded GitHub checkout steps to the pinned Node 24 release so routine CI
  and release logs no longer carry Node 20 deprecation warnings.
- Added persistent BuildKit, ccache, and Trivy database reuse across GitHub and
  Forgejo runners, including compiler cache mounts inside the ROCm Docker build.
- Kept periodic full model digest verification while caching unchanged results
  for seven days and using direct I/O on cache misses, preventing the 96 GiB
  checksum pass from consuming UMA and disabling monolithic DSpark validation.

### Added

- **release:** automate gfx1151 certification (`62205ed3`)
- **release:** automate certified promotion (`324956c7`)
- **ci:** automate candidate release chain (`a44a8205`)
- **ci:** mirror GitHub issues to Forgejo (`ad46d678`)
- **release:** finish deployment controls and XDNA docs (`4c37043a`)
- **xdna:** validate asynchronous DSpark provider (`1b439f8f`)
- **xdna:** schedule weighted expert runlists (`666c612d`)
- **xdna:** add async DSpark draft provider seam (`ac541642`)
- **xdna:** package ROCMFP2 XRT prototype (`e6f98a8d`)
- **engine:** prototype XDNA2 expert offload (`d41d2642`)

### Fixed

- **release:** compare sorted metadata paths (`fff01830`)
- **ci:** mount Forgejo CA in job containers (`4f64a037`)
- **release:** quiesce production supervisor (`27a9effb`)
- **ci:** make IOMMU preflight explicit (`3803c075`)
- **release:** avoid stale README version (`9b918f43`)
- **ci:** isolate release note permissions (`39de69f2`)
- **ci:** use native Forgejo push mirror (`d0ddd613`)
- **engine:** make XDNA corpus target-exact (`7472ac92`)
- **engine:** make resident XDNA acceptance target-exact (`6d0cb138`)
- **engine:** classify custom clamp types (`83a7b839`)
- **engine:** isolate resident graphs and XDNA context (`115d3a79`)
- **xdna:** validate mmap-backed model path (`907d0064`)
- **ci:** let the vulnerability scanner reach its database (`0e57f256`)
- **docker:** declare the image source so GHCR links the package (`0c33af6b`)
- **ci:** give the BuildKit container a working network (`3df0939a`)
- **release:** publish the container image where the docs promise it (`3ab6fada`)

### Performance

- **release:** cache model verification and expose hits (`a629fb25`)
- **release:** cache builds and preserve UMA capacity (`f2627063`)
- **engine:** batch resident verification safely (`06d15d7b`)
- **xdna:** attribute resident pipeline phases (`5a3171c1`)
- **xdna:** gate resident promotion by confidence (`d7d604b0`)
- **xdna:** capture current design control (`56c25ad2`)
- **xdna:** pipeline resident draft proposals (`6f8ba53c`)
- **xdna:** vectorize draft routed experts on CPU (`0c98d313`)
- **xdna:** keep draft Q8 stages resident (`cdec0696`)
- **engine:** fuse DSpark context KV on GPU (`4cbba655`)
- **xdna:** unify draft projection overlay (`358e1794`)
- **xdna:** keep draft projections on fixed overlay (`732be598`)
- **xdna:** parallelize DSpark CPU reductions (`3b830cab`)
- **engine:** measure GPU DSpark preprojection (`54270615`)
- **engine:** stage DSpark main projection on GPU (`f9f9d141`)
- **xdna:** reuse weights across draft context (`4fc7f551`)
- **xdna:** measure task-blocked Q8 projections (`a449b3ec`)
- **xdna:** add compensated Q8 draft shared expert (`efad17c8`)
- **xdna:** mask routed rows in gen8 (`5c97cd24`)
- **xdna:** add five-row ROCMFP4 expert kernel (`6d3b7acb`)
- **engine:** measure fused verifier MoE budget (`e888ef1a`)
- **xdna:** batch shared experts in gen6 (`de55413a`)
- **xdna:** measure concurrent GPU load (`f6e91bd7`)
- **xdna:** validate zero-copy GPU buffers (`76b9d3cf`)
- **xdna:** batch gen5 draft blocks (`8729f21b`)
- **xdna:** fuse expert pipeline in gen5 (`252b813f`)
- **xdna:** expose provider phase timings (`aaa0cf49`)
- **xdna:** vectorize ROCMFP2 decode in gen4 (`e811dfde`)
- **xdna:** preserve projection outputs in fp32 (`2de48a4f`)
- **xdna:** prototype second-generation expert kernel (`4d3c55a6`)

### Changed

- **engine:** prune unsupported backends and prototypes (`eade92dd`)

### Build and CI

- **xdna:** refresh runtime for queued drafts (`452c2e7d`)

### Documentation

- **xdna:** pin gen43 benchmark candidate (`8d0c90fd`)
- **xdna:** record whole-draft contention results (`8a05387a`)
- **xdna:** record AIE-RT queue rejection (`571d5d8c`)
- **xdna:** record gen5 validation scope (`b50e00bb`)
- **ci:** record the self-hosted builder and its fork-PR constraint (`8b9c5f99`)

### Testing

- **json:** cover UTF-8 output boundary (`48e0ca53`)
- **xdna:** enforce resident speculation gate (`37f7f6bf`)
- **xdna:** reject bfp16 draft projections (`9a8b0647`)
- **xdna:** validate trained draft experts (`41454f7f`)

### Maintenance

- **ci:** move checkout actions to Node 24 (`7987558a`)
## 2026.8.10

- Prepared the first source release.
- Added a one-command Docker Compose workflow with resumable model download.
- Pinned the default model revision and added expected-size, free-space, and
  SHA-256 checks before startup.
- Added host/container preflight diagnostics, Compose health reporting, and
  read-only plus generation smoke tests.
- Added OCI image metadata and operations/upstream-release guidance.
- Split the container into a full-toolchain `dev` target and a minimal
  dependency-closure `release` target used by Compose.
- Added licensing, vendor provenance, security, support, and contribution policy.
- Synced the latest agent-progress work: repeated-call diagnostics, a
  result-based progress lease, empty/degenerate-turn telemetry, and detection
  of tool markup delivered as visible text.
- Added opt-in stateless automatic loop recovery and Compose environment
  controls for progress reporting and recovery thresholds.
- Hardened request parsing against embedded NULs, duplicate or mistyped fields,
  unsupported HTTP transfer framing, and excess headers.
- Made strict tool-schema number equality exact beyond IEEE-754's integer range
  and replaced direct POSIX interpretation of ECMA-262 regex patterns with a
  translated, fail-closed subset.
- Removed one-off development probes and made static analysis, exact-commit
  tests, and gfx1151 certification blocking image-release gates.
- Promoted the gfx1151 HIP GEMM batch sweep and recorded the successful
  two-session hardware acceptance result.
- Changed the default Compose path to pull an immutable GHCR image while
  retaining an explicit local source-build override.
- Added project and third-party license notices to the runtime image and made
  the vulnerability-reporting fallback actionable.
- Added GitHub-hosted CPU and large-disk container workflows, direct GHCR
  publishing, and a manual environment-gated gfx1151 certification workflow.
- Fixed the ROCm 7.14 runtime image assembly to package both the gfx1151
  rocBLAS KPACK and its architecture-specific Tensile metadata/code-object
  tree from the pinned toolchain.
- Normalized loader-reported `/lib` paths into merged `/usr` locations so the
  collected runtime can be copied into the Ubuntu 24.04 release stage.
- Hardened gfx1151 certification to verify the immutable model pair,
  separately require exercised DSpark and disk-cache paths, use locked memory,
  and verify live speculative decoding outside resident batching.
- Repointed first-run acquisition to the current published quant and DSpark
  drafter, made both artifacts mandatory and digest-locked, and removed support
  for substituting unverified model files.
