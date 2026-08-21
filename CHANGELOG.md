# Changelog

Ember uses calendar versions in `YEAR.MONTH.DAY` form without zero-padding.
Release notes record user-visible features, fixes, compatibility changes,
upgrade steps, and validated hardware. Git tags add a `v` prefix; for example,
version `2026.8.10` is tagged `v2026.8.10`.

`VERSION` is authoritative. Ember publishes at most one release per calendar
day; additional fixes remain on `main` until the next dated release rather than
using an ambiguous same-day suffix.

## Unreleased

- Added `--host` and `EMBER_HOST` so trusted container/Kubernetes gateways can
  reach Ember while preserving the unauthenticated loopback default.
- Added `EMBER_VERIFY_EXISTING_SHA256=0` for trusted immutable model stores;
  downloaded artifacts remain pinned and checksum-verified before promotion.
- Integrated and documented the opt-in CPU/GPU/XDNA2 resident DSpark
  prototype. The XDNA Compose overlay now selects the measured two-session
  proposal pipeline instead of the slower target-expert placement experiment.
- Kept heterogeneous inference experimental: the fixed throughput fixture
  passed, while capture-graph output equivalence remains a promotion blocker.

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
