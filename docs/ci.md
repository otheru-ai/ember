# CI/CD

Ember runs the same release gates in two places:

- `.forgejo/workflows/` keeps the internal source-of-truth repository gated.
- `.github/workflows/` runs on the synchronized GitHub repository, publishes
  `ghcr.io/otheru-ai/ember`, and coordinates target-hardware certification.

## What CI covers, and what it deliberately cannot

The primary CI runs the **GPU-free half** of the project: the same
`cmake -S . -B build && ctest` that works on any host, plus sanitizers, static
analysis and a coverage ratchet. That is the stub-backend pipeline —
template → encode → generate → detokenize → SSE — which is where essentially
all server logic lives.

The separate `container.yml` workflow builds through the full ROCm `dev` stage
and publishes the minimal `release` stage on `vYEAR.MONTH.DAY` tags or manual
dispatch. A tag-triggered build must exactly match the root `VERSION` file;
manual builds receive a non-release `dev-<commit>` image tag. Every publish
first checks out and tests the exact event SHA. Versioned tags additionally
require `EMBER_GFX1151_CERTIFIED_SHA` to name that same commit after the
differential validator has passed on target hardware.
The build target is pinned to `gfx1151`, so compilation needs the HIP toolchain
and substantial disk but does not need a GPU.

The hosted workflows cannot cover these checks by themselves:

| Out of scope | Why |
|---|---|
| End-to-end runtime validation | Needs exclusive access to a gfx1151 GPU and model weights; `gfx1151-certify.yml` runs it on the protected Halo runner. |
| Differential validator | Needs the GPU and the 85 GiB GGUF; `gfx1151-certify.yml` runs exact, batched, and optional DSpark validation. |

Target-hardware certification is therefore manually dispatched and approval
gated, while its checks are automated. The one ROCm failure mode hosted CI can
catch cheaply is source-list drift between the two hand-maintained CMake lists
— see `ci/check_invariants.py`.

## Jobs

Ordered cheapest-first so a break reports in seconds.

| Job | Gates? | What it catches |
|---|---|---|
| `invariants` | yes | A `src/` file in only one CMake list (the `d8ace73` bug class); a test that compiles but is never registered with ctest; a target that escapes `-Werror`. |
| `build-test` | yes | `EMBER_STRICT=ON` (warnings-as-errors) across Release **and** Debug, then the full 38-test suite. |
| `sanitizers` | yes | ASan + UBSan + LeakSanitizer over the whole suite. |
| `analyzer` | yes | New `gcc -fanalyzer` or `cppcheck` findings. |
| `coverage` | yes | Per-file line-coverage regression against `ci/coverage_floors.json`. |
| `source-gate` | release gate | Strict Release build and full GPU-free suite against the exact commit being published. |
| `release-image` | release gate | Requires gfx1151 certification for version tags, validates CalVer, builds the `dev` stage, extracts the minimal runtime closure, emits SBOM/provenance, pushes version plus commit tags, reports image size, and rejects fixed critical vulnerabilities. |
| `certify-gfx1151` | manual hardware gate | Verifies the immutable image and model digests, GEMM batches, exact and resident-session differential paths, optional DSpark, and a live generation request on Strix Halo. |

## Runner setup

### GitHub-hosted runners and GHCR

`.github/workflows/ci.yml` uses the standard `ubuntu-24.04` hosted runner. It
needs no repository secrets and covers invariants, strict Debug/Release builds,
all tests, sanitizers, both analyzers, and the coverage ratchet.

The ROCm image needs more than the standard runner's 14 GB disk, so
`release-image` runs on `${{ vars.EMBER_BUILD_RUNNER || 'ubuntu-latest-8-cores' }}`.

GitHub larger runners require a **Team or Enterprise Cloud plan**. The
`otheru-ai` organization is currently on the free plan, where
`ubuntu-latest-8-cores` can never be scheduled: the job queues silently and
GitHub cancels it after 24 hours. That is precisely how the `v2026.8.10` publish
was lost -- `source-gate` passed and `release-image` was cancelled a day later,
so no image reached GHCR. Pick one:

- **Upgrade the organization to Team**, then leave `EMBER_BUILD_RUNNER` unset so
  the default larger-runner label applies. Grant the Ember repository access to
  the runner.
- **Stay on the free plan** and set the repository variable
  `EMBER_BUILD_RUNNER` to a self-hosted builder label with Docker, Buildx and at
  least 200 GiB free. The `dev` stage alone is ~21 GB, so a standard
  `ubuntu-24.04` runner cannot substitute even with aggressive disk reclaim.

The second option is what Ember currently uses. The builder is registered as the
repository runner `ember-builder-halo` with the label `ember-builder`, and
`EMBER_BUILD_RUNNER` is set to that label. It runs as a systemd **user** service
(`systemctl --user status actions-runner-ember`) with lingering enabled, so it
survives logout and reboot without a root-owned unit.

Because Ember is a public repository, a fork pull request can edit a workflow to
target the builder's label. Repository Actions settings therefore require
approval for **all external contributors** before any fork workflow runs; do not
relax that while a self-hosted runner is attached.

The job also carries `timeout-minutes: 180` so a misconfigured runner fails the
same day instead of consuming the full 24-hour queue ceiling.

The container workflow uses only GitHub's scoped `GITHUB_TOKEN` with
`packages: write`; no long-lived GHCR token is stored in GitHub.

Run the Container workflow manually on a candidate commit before certification.
It publishes `dev-<sha12>` and `sha-<sha12>` images with SBOM/provenance, then
reports size and runs the critical-vulnerability gate. After the first publish,
make the `otheru-ai/ember` GHCR package public so Compose can pull it
anonymously. GitHub exposes no REST endpoint for this; it is a one-time manual
step under Package settings -> Danger Zone -> Change visibility.

For the hardware gate, register the Halo host as a GitHub self-hosted runner
with labels `self-hosted`, `linux`, `x64`, and `gfx1151`. Create a GitHub
environment named `gfx1151-certification` with required reviewers. The workflow
is manual-only and never runs for a pull request. Before approving it, schedule
a maintenance window and stop any server already occupying the GPU; the
workflow deliberately refuses to stop operator services itself.

The certification sequence is:

1. Manually run **Container** for the candidate SHA.
2. Run **gfx1151 certification** with that full SHA, the pinned quant and
   drafter paths, a short non-sensitive prompt that produces enough output to enter
   DSpark, a separate non-sensitive prompt of at least 512 model tokens for the
   disk round trip. The workflow itself pins and verifies both published
   digests; they cannot be substituted at dispatch time.
3. After the validators and generation smoke test pass, set the GitHub
   repository variable `EMBER_GFX1151_CERTIFIED_SHA` to the full SHA.
4. Create and mirror the matching `vYEAR.MONTH.DAY` tag. The tag workflow
   refuses any other SHA and publishes the CalVer and `latest` GHCR tags.

The self-hosted runner needs Docker, `curl`, Python 3, `/dev/kfd`, `/dev/dri`,
and read access to the model and validation prompt. The workflow pulls the
already-built `sha-*` image, verifies its full OCI revision label, mounts model
files read-only, uses disposable Docker volumes for KV state, and never checks
out repository content onto the Halo runner.

### Forgejo runners

CI needs one Forgejo runner with the `docker` label. On a host with Docker:

```bash
# 1. In Forgejo: Site Administration -> Actions -> Runners -> Create new runner
#    (or per-repo: Settings -> Actions -> Runners). Copy the registration token.

# 2. On the runner host:
forgejo-runner register \
  --no-interactive \
  --instance https://forge.example.com \
  --token <REGISTRATION_TOKEN> \
  --name ember-ci \
  --labels docker:docker://node:20-bookworm

forgejo-runner daemon
```

`forgejo-runner list` prints the labels a runner offers. If yours registered
under a different label, change `runs-on:` in `.forgejo/workflows/ci.yml` to
match — that is the single most common reason a workflow sits queued forever.

Actions must also be enabled for the repository: **Settings → Advanced →
Enable Repository Actions**.

Container publishing needs a separate host runner labeled `docker-build` with
Docker Buildx and at least 200 GiB free. Configure these repository values:

| Name | Kind | Example |
|---|---|---|
| `EMBER_REGISTRY` | variable | `ghcr.io` |
| `EMBER_IMAGE` | variable | `otheru-ai/ember` |
| `EMBER_GFX1151_CERTIFIED_SHA` | variable | Full commit SHA that passed target validation |
| `REGISTRY_USERNAME` | secret | registry service account |
| `REGISTRY_TOKEN` | secret | token with image push permission |

For Forgejo-driven GHCR publishing, use `ghcr.io` and `otheru-ai/ember`; the token
needs `write:packages`. This token is needed only by Forgejo—GitHub Actions uses
its built-in `GITHUB_TOKEN`. Keep the immutable CalVer and `sha-*` tags even
though the workflows also update `latest` for discovery.

The internal Forgejo repository remains the source of truth. Configure its push
mirrors in repository settings for each public GitHub, GitLab, or Gitea target;
use a dedicated token with repository-write permission, mirror only `main` and
release tags, and enable synchronization after pushes. Mirror credentials stay
in Forgejo and never enter this repository.

The builder never needs `/dev/kfd`, `/dev/dri`, or model weights. The separate
`gfx1151` runner pulls the immutable commit image and performs the hardware
checks before a versioned release is allowed.

The workflow uses `actions/checkout@v4`, which Forgejo resolves against
`code.forgejo.org` by default (`[actions] DEFAULT_ACTIONS_URL`). If the runner
has no egress there, mirror the action locally or replace the step with a plain
`git clone`.

## The coverage ratchet

Aggregate coverage hides the regressions worth catching — a new untested
400-line module barely moves an 80% total. So floors are **per file**, checked
in at `ci/coverage_floors.json`:

```bash
# report + enforce (what CI runs)
python3 ci/coverage.py --build build-cov

# re-baseline after adding tests; only ever raises a floor
python3 ci/coverage.py --build build-cov --update
```

`--update` refuses to lower a floor, so a commit that loses coverage cannot
quietly rewrite the baseline. Lowering one is a deliberate edit of the JSON and
shows up in review.

## Reproducing a CI failure locally

```bash
# build-test
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DEMBER_STRICT=ON
cmake --build build -j"$(nproc)" && ctest --test-dir build --output-on-failure

# sanitizers  (on WSL2, prefix ctest with `setarch -R` — see below)
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j"$(nproc)"
ASAN_OPTIONS=detect_leaks=1 ctest --test-dir build-asan --output-on-failure

# invariants
python3 ci/check_invariants.py
```

## Sanitizers on WSL2 — disable ASLR or the results are noise

On WSL2 (`*-microsoft-standard-WSL2`), ASan's shadow-memory reservation
intermittently collides with the kernel's ASLR layout. The process then spins
printing `AddressSanitizer:DEADLYSIGNAL` until something kills it. Measured on
this repo's own suite:

| | result |
|---|---|
| plain `ctest` | 7 tests failed, **a different set each run**, 523 s wall |
| `test_background_gate` alone (18 lines, 0.00 s) × 15 | **6 hung** |
| `setarch -R` + same binary × 15 | **15 passed** |
| `setarch -R` + full suite | **all tests passed, 1.8 s** |

So on WSL2, always:

```bash
setarch -R env ASAN_OPTIONS=detect_leaks=1 \
  ctest --test-dir build-asan --output-on-failure
```

Two traps worth knowing:

- **The failures look like hangs in random tests.** Because the failing set
  changes every run and includes tests that cannot possibly hang, it is easy to
  misread as contention and dismiss. If the set is not stable, suspect the
  environment before the code.
- **LeakSanitizer fails silently in the other direction too.** Without
  `setarch -R` it also *missed* a real leak that it reports reliably with ASLR
  off. A green local ASan run on WSL2 proves nothing either way.

CI is unaffected — Forgejo runners execute in normal Linux containers, so the
`sanitizers` job needs no workaround.
