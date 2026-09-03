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

After every GitHub `main` push passes all CI jobs, CI calls `container.yml` and
automatically builds the full ROCm `dev` stage and publishes the minimal
`release` stage as immutable `dev-<commit>` and `sha-<commit>` candidates.
`container.yml` also runs directly on `vYEAR.MONTH.DAY` tags or manual
dispatch. A tag-triggered build must exactly match the root `VERSION` file;
every publish checks out and tests the exact event SHA. Versioned tags
must point to a metadata-only release commit whose parent is named by
certification from this repository's Actions run history after the differential validator has passed on
target hardware. This keeps every executable input identical to the certified
candidate while allowing automation to update the version and notes.
The build target is pinned to `gfx1151`, so compilation needs the HIP toolchain
and substantial disk but does not need a GPU.

The hosted workflows cannot execute these checks by themselves:

| Out of scope | Why |
|---|---|
| End-to-end runtime validation | Needs exclusive access to a gfx1151 GPU and model weights; `gfx1151-certify.yml` runs it on the dedicated Halo runner. |
| Differential validator | Needs the GPU and the 85 GiB GGUF; `gfx1151-certify.yml` runs exact, batched, and optional DSpark validation. |

Target-hardware certification starts automatically after the immutable
candidate passes the hosted and container gates. The one ROCm failure mode
hosted CI can catch cheaply is source-list drift between the two hand-maintained
CMake lists — see `ci/check_invariants.py`.

The saved-ISA ROCMI4 W4A8 compile-evidence gate is intentionally GitHub-only.
It builds both packing variants in AMD's pinned ROCm 10.0 development container
and retains their assembly, object, disassembly, and CMake contract as GitHub
release evidence. Every change below `engine/ggml/src/ggml-cuda/` or
`engine/ggml/rocmfpx/` triggers it. Forgejo remains the source and GPU-free CI
gate, but does not duplicate this multi-gigabyte ROCm artifact job: the mirrored
commit cannot enter GitHub publication or gfx1151 certification until GitHub's
compile-evidence workflow passes. This is an intentional artifact-retention and
publisher boundary, not a claim that the Forgejo CPU suite proves the production
HIP translation unit.

## Jobs

Ordered cheapest-first so a break reports in seconds.

| Job | Gates? | What it catches |
|---|---|---|
| `invariants` | yes | A `src/` file in only one CMake list (the `d8ace73` bug class); a test that compiles but is never registered with ctest; a target that escapes `-Werror`. |
| `build-test` | yes | `EMBER_STRICT=ON` (warnings-as-errors) across Release **and** Debug, then the full test suite. |
| `sanitizers` | yes | ASan + UBSan + LeakSanitizer over the whole suite. |
| `analyzer` | yes | New `gcc -fanalyzer` or `cppcheck` findings. |
| `coverage` | yes | Per-file line-coverage regression against `ci/coverage_floors.json`. |
| `source-gate` | release gate | Strict Release build and full GPU-free suite against the exact commit being published. |
| `publish-candidate` | release-candidate gate | After every `main` CI job passes, calls the container workflow for that exact SHA. |
| `release-image` | release gate | Publishes immutable commit candidates automatically; for version tags it verifies the metadata-only child of the certified gfx1151 tree, validates CalVer, pushes version and `latest`, and rejects fixed critical vulnerabilities. |
| `certify-and-release` | release gate | Calls the dedicated Strix Halo runner after candidate publication; it verifies the immutable image and model digests, GEMM batches, exact and resident-session differential paths, DSpark, and a live generation request, then promotes the candidate. |

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

Release builds reuse five cache layers. Hosted CPU jobs restore ccache
objects by compiler/configuration; the Forgejo containers share the named
`ember-ci-ccache` volume; the self-hosted image builder reuses one persistent
BuildKit instance plus the Dockerfile's 20 GiB ccache mount; and both image
publishers retain Trivy's database in `ember-trivy-cache`. The Halo runner also
caches successful model-integrity checks for seven days, invalidating them on
any device, inode, size, mtime, or ctime change. Keep
`EMBER_BUILDX_BUILDER` stable across runs—creating a run-specific builder
strands the expensive ROCm layers in an unreachable BuildKit volume. GitHub
serializes candidate and tag builds because they share that instance.

Certification deliberately does not reuse runtime KV state. Model digests are
rechecked after seven days or immediately when filesystem identity changes.
Cache misses use XFS direct I/O: buffered hashing followed by
`POSIX_FADV_DONTNEED` was measured to leave only 24 GiB available on the Halo
host, whereas HIP's monolithic placement needs at least 100 GiB. Direct I/O
keeps the 96 GiB verification pass out of the shared CPU/GPU page cache.
The gfx1151 runner controls production through the root-owned
`/usr/local/sbin/ember-cert-production` wrapper. Its sudo policy permits only
`is-active`, `stop`, and `start` for `ember-server.service`; stopping only the
child container is incorrect because the root user service immediately
recreates it.

Because Ember is a public repository, a fork pull request can edit a workflow to
target the builder's label. Repository Actions settings therefore require
approval for **all external contributors** before any fork workflow runs; do not
relax that while a self-hosted runner is attached.

The job also carries `timeout-minutes: 180` so a misconfigured runner fails the
same day instead of consuming the full 24-hour queue ceiling.

The container workflow uses only GitHub's scoped `GITHUB_TOKEN` with
`packages: write`; no long-lived GHCR token is stored in GitHub.

Every mirrored `main` commit automatically enters Container after the complete
GitHub CI matrix passes. It publishes `dev-<sha12>` and `sha-<sha12>` images
with SBOM/provenance, then reports size and runs the critical-vulnerability
gate. Manual dispatch remains available to retry a failed infrastructure build
without creating a new commit. After the first publish, make the
`otheru-ai/ember` GHCR package public so Compose can pull it anonymously.
GitHub exposes no REST endpoint for this; it is a one-time manual step under
Package settings -> Danger Zone -> Change visibility.

The hardware gate uses the dedicated repository runner
`ember-gfx1151-prod`, registered on the Halo host with labels `self-hosted`,
`linux`, `x64`, and `gfx1151`. Only a trusted push to `main` can call it; pull
request jobs never target this runner. Certification checks IOMMU and device
access, generates deterministic non-sensitive prompts in the runner temporary
directory, stops the configured production container for exclusive GPU access,
and restores it even when a validator fails. Manual dispatch remains available
only for infrastructure recovery or an explicit CalVer override.

The certification sequence is:

1. Push the candidate to Forgejo `main`. The mirror, GitHub CI, immutable
   candidate-image publication, vulnerability scan, and gfx1151 certification
   happen automatically.
2. The hardware job reads the fixed quant and drafter paths from repository
   variables, verifies both published digests, and uses generated prompts for
   DSpark and the disk round trip. Neither model can be substituted at dispatch
   time.
3. After the validators and generation smoke test pass, the promotion job uses
   the current UTC date (or the optional dispatch `release_version`) to generate
   a grouped changelog, update `VERSION` and the Compose image pin, and
   create an annotated tag on that metadata-only child commit. It aborts if
   `main` advanced during certification or the date has already been released.
4. The promotion job records the certified parent SHA on GitHub and atomically
   pushes the release commit plus tag to Forgejo. The native mirror carries both
   to GitHub. The tag workflow verifies the parent and the exact three-file
   allowlist, publishes the CalVer and `latest` GHCR tags, then creates the
   GitHub release from the new `CHANGELOG.md` section.

The self-hosted runner needs Docker, `curl`, Python 3, `/dev/kfd`, `/dev/dri`,
an enabled IOMMU, and read access to the model pair. The workflow pulls the
already-built `sha-*` image, verifies its full OCI revision label, mounts model
files read-only, uses disposable Docker volumes for KV state, and never checks
out repository content onto the Halo runner.

The pinned files are the target and DSpark draft from
[`otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF`](https://huggingface.co/otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF).
The Halo host keeps shorter internal filenames, but certification verifies the
published SHA-256 digests (`a936e0a5…3d54` and `1a01c80e…ae6`) before either
file reaches the engine.

### Forgejo runners

CI needs one repository-scoped Forgejo runner with `docker` and `docker-build`
labels. Ember's runner is a persistent systemd user service; its `docker` label
uses container isolation and permits only the `ember-ci-ccache` named volume,
while the manually dispatched recovery publisher uses the trusted
`docker-build:host` label. On the current WSL/Docker 29 builder, configure
`container.network: host`; Docker bridge creation returns `operation not
supported`, while host networking preserves disposable container isolation and
lets the runner cache endpoint resolve reliably. Because Forgejo uses an
internal CA, each workflow container also mounts only the pinned root
certificate read-only; add its exact source path to `container.valid_volumes`.
On a host with Docker:

```bash
# 1. In Forgejo: Site Administration -> Actions -> Runners -> Create new runner
#    (or per-repo: Settings -> Actions -> Runners). Copy the registration token.

# 2. On the runner host:
forgejo-runner register \
  --no-interactive \
  --instance https://forge.example.com \
  --token <REGISTRATION_TOKEN> \
  --name ember-ci \
  --labels docker:docker://node:24-bookworm,docker-build:host

forgejo-runner daemon
```

`forgejo-runner list` prints the labels a runner offers. If yours registered
under a different label, change `runs-on:` in `.forgejo/workflows/ci.yml` to
match — that is the single most common reason a workflow sits queued forever.

Actions must also be enabled for the repository: **Settings → Advanced →
Enable Repository Actions**.

### Automatic Forgejo-to-GitHub release chain

Forgejo's native push mirror sends source-of-truth refs to
`https://github.com/otheru-ai/ember.git`. Configure it under repository
**Settings → Repository → Push Mirrors** with synchronization after commits
enabled. Use a fine-grained GitHub token restricted to `otheru-ai/ember` with
repository **Contents: read and write**. The credential remains in Forgejo's
encrypted mirror configuration and never enters this repository or a runner.

The mirror is also reconciled every eight hours so a transient GitHub outage
does not require another source commit. Forgejo currently has only the `main`
branch; treat any future branch pushed there as public because the native
mirror publishes repository refs, not a private allowlist.

The resulting GitHub `main` push starts `.github/workflows/ci.yml`. Its
`publish-candidate` job waits for invariants, both strict builds, sanitizers,
analyzers, and coverage, then calls the reusable Container workflow. GitHub's
built-in `GITHUB_TOKEN` publishes `dev-<sha12>` and `sha-<sha12>` to GHCR. A
release-metadata commit skips a redundant candidate build. Its mirrored version
tag starts Container directly; the certified-parent, three-file allowlist, and
VERSION checks must pass before the CalVer and `latest` tags are written.

The promotion job runs on `ember-builder` after the hardware job.
Configure these GitHub repository values in addition to the runner label:

| Name | Kind | Purpose |
|---|---|---|
| `FORGEJO_RELEASE_SSH_KEY` | secret | Private half of a repository-scoped, write-enabled Forgejo deploy key. |
| `FORGEJO_SSH_HOST_KEYS` | secret | Pinned SSH host keys for the source-of-truth Forgejo endpoint. |
| `FORGEJO_RELEASE_REMOTE` | variable | Source-of-truth SSH URL, currently `ssh://git@git.otheru.ai:2222/otheru/ember.git`. |
| `EMBER_BUILDX_BUILDER` | variable | Stable Buildx instance on `ember-builder`; defaults to `ember-release`. Never generate this per run. |
| `EMBER_GFX1151_CERTIFIED_SHA` | variable | Forgejo's manual disaster-recovery publisher only. The GitHub release path reads certification from this repository's Actions run history instead, so it needs no credential and nothing to rotate. |
| `EMBER_CERT_MODEL_PATH` | variable | Absolute Halo-host path to the pinned target GGUF. |
| `EMBER_CERT_DRAFT_PATH` | variable | Absolute Halo-host path to the pinned DSpark draft GGUF. |
| `EMBER_CERT_PORT` | variable | Unused loopback port for the certification server; defaults to `18080`. |
| `EMBER_PRODUCTION_CONTAINER` | variable | Container quiesced and restored around certification; defaults to `ember-server`. |

Keep the deploy key write-enabled only for this repository. Never commit its
private half or the GitHub automation token. The generated release commit is
deliberately limited to `CHANGELOG.md`, `VERSION`, and `compose.yaml`; any other
path makes both promotion and publication fail closed.

`.forgejo/workflows/container.yml` remains a manually dispatched
disaster-recovery publisher; it no longer reacts to tags, preventing Forgejo
and GitHub builders from racing to update the same GHCR tags.

Container publishing needs a separate host runner labeled `docker-build` with
Docker Buildx and at least 200 GiB free. Configure these repository values:

| Name | Kind | Example |
|---|---|---|
| `EMBER_REGISTRY` | variable | `ghcr.io` |
| `EMBER_IMAGE` | variable | `otheru-ai/ember` |
| `EMBER_GFX1151_CERTIFIED_SHA` | variable | Forgejo break-glass publisher only; set by hand. GitHub proves certification from run history. |
| `REGISTRY_USERNAME` | secret | registry service account |
| `REGISTRY_TOKEN` | secret | token with image push permission |

For Forgejo-driven GHCR publishing, use `ghcr.io` and `otheru-ai/ember`; the token
needs `write:packages`. This token is needed only by Forgejo—GitHub Actions uses
its built-in `GITHUB_TOKEN`. Keep the immutable CalVer and `sha-*` tags even
though the workflows also update `latest` for discovery.

The internal Forgejo repository remains the source of truth. Public GitHub
branches and release tags are outputs of the workflow above; do not commit on
the GitHub mirror or configure a reverse code mirror.

The builder never needs `/dev/kfd`, `/dev/dri`, or model weights. The separate
`gfx1151` runner pulls the immutable commit image and performs the hardware
checks before a versioned release is allowed.

The workflow uses `actions/checkout@v5`, which Forgejo resolves against
`code.forgejo.org` by default (`[actions] DEFAULT_ACTIONS_URL`). If the runner
has no egress there, mirror the action locally or replace the step with a plain
`git clone`.

## Mirroring GitHub issues into Forgejo

Issues arrive on the published GitHub repository, but the source of truth is
Forgejo — and issues are not git objects. No remote configuration will ever
carry them across; they are rows in a forge database, reachable only through
its API.

Forgejo cannot pull them either. Marking a repository as a mirror greys out the
issue/label/milestone migration options ([forgejo#4962]), and the request to
sync them on an interval ([forgejo#5367]) is open with no implementation —
"syncing issues & discussions will [be] some nice feature down the roadmap".
Upstream Gitea has carried the same request since 2017 ([gitea#1876]). The
off-the-shelf tool for this, [gitea-mirror], is built to create and manage
*pull mirrors*, which is the wrong topology here: `otheru/ember` on Forgejo is
`origin`, not a downstream copy, and turning it into a pull mirror would make
it read-only.

So `.github/workflows/mirror-issues.yml` does the copy, running
`ci/mirror_gh_issues.py` on the `ember-builder` runner. That host is already on
the tailnet `git.otheru.ai` lives on, so the sync reaches **out** to Forgejo and
GitHub never needs a route in — no webhook, no public ingress. That is what
makes it event-driven instead of polled; the hourly `schedule` only reconciles
what arrived while the runner was offline.

The mapping lives in the mirrored text, not a database: every issue and comment
the script writes ends with an `<!-- ember-gh-mirror:issue:N -->` marker, so a
run re-derives the whole mapping by reading Forgejo back. Runs are idempotent
and the runner holds no state worth backing up.

### Setup

1. Create the bot account that will author mirrored issues. Forgejo cannot
   attribute an issue to a GitHub user, so every mirrored issue is authored by
   whoever owns the token; a dedicated account keeps that legible instead of
   making a human appear to have filed reports they never saw.

   - Site Administration → Identity & Access → Accounts → Create User:
     username `github-mirror`, email `github-mirror@otheru.ai`, a generated
     password. **Clear "require password change"** — a pending forced change
     rejects API authentication, which surfaces as an unexplained 403.
   - Sign in as `github-mirror`, then Settings → Applications → Generate Token
     with **`write:issue`** scope only. Nothing else is needed: the mirror never
     reads user data, never touches code, and never writes to GitHub.
   - Grant it write access to the repository:

     ```bash
     curl -fsS --cacert /home/mythos/.config/ember/forgejo-root.crt \
       -H "Authorization: token $FORGEJO_ADMIN_TOKEN" \
       -X PUT -H 'Content-Type: application/json' -d '{"permission":"write"}' \
       https://git.otheru.ai/api/v1/repos/otheru/ember/collaborators/github-mirror
     ```

   The token must be minted while signed in *as* the bot. There is no admin
   shortcut: Forgejo exposes no `/admin/users/{username}/tokens` endpoint, and
   `/users/{username}/tokens` requires BasicAuth precisely so that a stolen API
   token cannot mint credentials for other accounts.

   If the instance authenticates through Authentik and local password login is
   disabled, create the bot in Authentik instead and let it provision through —
   the token step is the same once you can sign in as it.
2. In the GitHub repository, add secrets `FORGEJO_TOKEN` and `FORGEJO_URL`
   (`https://git.otheru.ai`), and variables `FORGEJO_REPO` (`otheru/ember`) and
   `FORGEJO_CA_BUNDLE`.

   `FORGEJO_URL` is a **secret rather than a variable** even though a hostname
   is not a credential. Ember is a public repository, so its Actions logs are
   world-readable, and `vars` are echoed into them unredacted while secrets are
   masked. `git.otheru.ai` resolves only inside the tailnet — it is NXDOMAIN in
   public DNS — so leaking it into a log would publish infrastructure naming
   that is otherwise unlisted. The script reinforces this by logging API paths
   without their host (see `Forge.name`); the secret is the second layer, not
   the only one.

   Secrets themselves are never public: they are encrypted at rest, cannot be
   read back through the UI or API, and are withheld from workflows triggered by
   `pull_request` from a fork. The trust boundary that matters is write access —
   anyone who can push a workflow can print a secret in transformed form and
   defeat masking.
3. Place the internal root CA on the runner host and set the `FORGEJO_CA_BUNDLE`
   variable to its path. Caddy issues `git.otheru.ai` a 12-hour certificate from
   `CN = Caddy Local Authority - 2026 ECC Root`, which is in no default trust
   store, so without this every request fails verification. The root is not
   recoverable from the TLS handshake — Caddy presents leaf and intermediate
   only, which is correct behaviour and precisely why a trust anchor has to
   arrive out of band.

   Fetch it from the internal distribution point, which is deliberately
   unauthenticated — note `auth.otheru.ai/ca/`, **not**
   `cert.otheru.ai/otheru-root-ca.crt`; the latter sits behind the Authentik
   forward-auth proxy and returns a 302 to a login page rather than a
   certificate, which cannot bootstrap trust for an unattended runner:

   ```bash
   curl -fsSL https://auth.otheru.ai/ca/otheru-root-ca.crt \
     -o /tmp/otheru-root-ca.crt
   install -Dm644 /tmp/otheru-root-ca.crt \
     /home/mythos/.config/ember/forgejo-root.crt
   ```

   That first `curl` is itself unverifiable — the distribution point is served
   under the very CA being fetched — so confirm the anchor rather than trusting
   the download. A CA that does not chain to this leaf fails identically to no
   CA at all, and `Verification: OK` is the only acceptable output:

   ```bash
   openssl s_client -connect git.otheru.ai:443 -servername git.otheru.ai \
     -CAfile /home/mythos/.config/ember/forgejo-root.crt \
     -verify_return_error </dev/null 2>&1 | grep Verification
   ```

   As of 2026-08-16 that anchor is `CN = Caddy Local Authority - 2026 ECC Root`,
   SHA-256 fingerprint
   `D0:8C:5F:83:20:8A:0F:2E:38:65:20:2C:00:2A:5C:D1:EA:1A:10:C8:A0:F0:36:95:F8:71:80:A0:B7:9B:F3:84`,
   valid until 2035-12-11. The pin is to that root, which outlives the 12-hour
   leaf; if it is ever rotated the file must be replaced, and the failure is
   loud and names `FORGEJO_CA_BUNDLE`.

Verify without writing anything, from the runner host:

```bash
GITHUB_TOKEN="$(gh auth token)" GITHUB_REPOSITORY=otheru-ai/ember \
FORGEJO_URL=https://git.otheru.ai FORGEJO_REPO=otheru/ember \
FORGEJO_TOKEN=... FORGEJO_CA_BUNDLE=/home/mythos/.config/ember/forgejo-root.crt \
python3 ci/mirror_gh_issues.py --all --dry-run
```

`FORGEJO_INSECURE=1` skips certificate verification. It exists so a first-run
operator can prove the rest of the pipeline works before wiring up the CA; it
prints a warning and should never be set in the workflow.

### Constraints worth knowing

- **One way, always.** Nothing writes back to GitHub. Comments added on the
  Forgejo copy go nowhere, which the mirrored issue header says explicitly.
- **Authorship is the token's.** Forgejo has no way to attribute an issue to a
  GitHub account, so mirrored issues are authored by `github-mirror` and the
  original author and a link lead the body. Changing the token later does not
  re-attribute issues already mirrored.
- **`#123` and `@name` inside a mirrored body resolve against Forgejo**, not
  GitHub. Rewriting them would corrupt quoted text, so they are left alone.
- **Deletions do not propagate.** A GitHub issue deleted outright leaves its
  Forgejo copy behind; there is no event to observe and nothing to fetch.
- **Pull requests are skipped.** GitHub's issues API returns PRs, and the
  `issue_comment` event fires for PR comments; both are filtered.
- Scheduled workflows are disabled by GitHub after 60 days of repository
  inactivity. A manual dispatch re-arms them.
- Because the builder is self-hosted on a public repository, **do not add a
  `pull_request` trigger to this workflow.** `issues` and `issue_comment` always
  execute the default-branch definition, so no fork-authored code runs; a
  `pull_request` trigger would break that and hand a fork the Forgejo token's
  runner. Issue titles and bodies are attacker-controlled text, so the workflow
  interpolates nothing from `github.event` into `run:` — the script reads the
  payload file itself and trusts only the integer issue number.

[forgejo#4962]: https://codeberg.org/forgejo/forgejo/issues/4962
[forgejo#5367]: https://codeberg.org/forgejo/forgejo/issues/5367
[gitea#1876]: https://github.com/go-gitea/gitea/issues/1876
[gitea-mirror]: https://github.com/RayLabsHQ/gitea-mirror

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
