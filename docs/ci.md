# CI/CD

Ember's reference CI uses Forgejo Actions' GitHub-Actions-compatible syntax,
with workflows in `.forgejo/workflows/`.

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

The workflows cannot cover:

| Out of scope | Why |
|---|---|
| End-to-end runtime validation | Needs exclusive access to a gfx1151 GPU and model weights. |
| Differential validator | Needs the GPU and the 85 GiB GGUF. |

Those stay manual and are listed in `AGENTS.md`. The one ROCm failure mode CI
*can* catch cheaply is source-list drift between the two hand-maintained CMake
lists — see `ci/check_invariants.py`.

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
| `release-image` | release gate | Requires gfx1151 certification for version tags, validates CalVer, builds the `dev` stage, extracts the minimal runtime closure, emits SBOM/provenance, and pushes version plus commit tags. |

## Runner setup

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
| `EMBER_REGISTRY` | variable | `registry.example.org` |
| `EMBER_IMAGE` | variable | `otheru/ember` |
| `EMBER_GFX1151_CERTIFIED_SHA` | variable | Full commit SHA that passed target validation |
| `REGISTRY_USERNAME` | secret | registry service account |
| `REGISTRY_TOKEN` | secret | token with image push permission |

The builder never needs `/dev/kfd`, `/dev/dri`, or model weights. A future
self-hosted runner labeled `gfx1151` should pull the immutable commit tag and
run the differential validator plus `scripts/smoke_test.sh --generate` before a
release digest is promoted.

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
