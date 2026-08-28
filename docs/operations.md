# Operations guide

This guide covers the release-default DeepSeek-V4-Flash ROCMFPx deployment and
the explicit, candidate-only Qwen3.8-Flash-Next deployment boundary on a native
Linux AMD Strix Halo (`gfx1151`) host. Ember is not a general-purpose GGUF
runner, and the release image is not expected to work on other GPU or NPU
architectures.

## Host prerequisites

- x86_64 Linux; Docker Desktop and virtualized GPU passthrough are unsupported.
- A ROCm-supported kernel/device stack exposing `/dev/kfd` and `/dev/dri`.
- AMD Strix Halo (`gfx1151`). The image is compiled specifically for this ISA.
- Approximately 128 GiB unified memory. The tested model is 85.3 GiB and uses
  about 89 GiB resident before context and operating-system overhead.
- At least 100 GiB free in the model directory for the verified pair. The final
  artifacts occupy about 95.4 GiB together (91,547,243,200 bytes for the quant
  and 10,897,111,840 bytes for the drafter); 120 GiB free is recommended so a
  resumed `.part` file, checksum pass, and filesystem overhead do not consume
  the last margin. The preflight script enforces the 100 GiB minimum.
- Docker Engine with the Compose v2 plugin.

The normal release needs no NPU software. The opt-in heterogeneous image also
requires the host `amdxdna` driver and compatible firmware, translated IOMMU
domains (do not boot with `amd_iommu=off`), render-group access, and a visible
`/dev/accel/accel0`. The image supplies XRT userspace; it cannot replace the
host kernel module or firmware.

Run the host checks before the first build:

```bash
scripts/preflight.sh
```

Warnings are advisory; missing devices, Docker access, architecture, or disk
space are errors. The container repeats the device and architecture checks
before it downloads model data.

## Start and readiness

```bash
docker compose up -d
```

The first invocation pulls the immutable target-specific image, downloads the
pinned quant and DSpark drafter with resumable HTTP, verifies both SHA-256
digests, loads them, and starts the API on `http://127.0.0.1:8080`. `./models`
and `./cache` persist across container replacement. Follow progress with
`docker compose logs -f ember`.

The deployable image contains the stripped server, crash shim, recursive
dynamic-library closure, rocBLAS runtime kernel data, and license notices.
Source, headers, compilers, build intermediates, and model weights are absent
from its filesystem. To build that image locally instead of pulling it:

```bash
docker compose -f compose.yaml -f compose.build.yaml up --build -d
```

The full ROCm SDK and compiler exist only in the Dockerfile's `dev` stage.

Compose reports the service healthy only after `GET /health` succeeds. The
20-minute health start period accommodates first load on the release hardware;
it does not bypass Ember's own startup failures.

For subsequent starts:

```bash
docker compose up -d
scripts/smoke_test.sh
```

The smoke test checks health, model discovery, and status without generating
tokens. Use `scripts/smoke_test.sh --generate` after an upgrade to exercise a
short end-to-end inference request.

## Opt-in Qwen3.8-Flash-Next deployment

Qwen deployment is explicit and local-artifact-only. The entrypoint does not
fall back to the DeepSeek drafter, infer a speculative depth, or download an
unpublished candidate. Obtain one certified candidate bundle containing:

- the complete ordered main GGUF shard set;
- its matching quantized MTP GGUF and selected depth from the bakeoff record;
- `Qwen3.8-Flash-Next-BF16-mmproj.gguf`;
- `Qwen3.8-Flash-Next-vocab-only.gguf`; and
- a basename-only `SHA256SUMS` whose release evidence also gives the checksum
  file's own SHA-256.

All of those model artifacts and `SHA256SUMS` must be regular, non-symlink
files directly beneath the directory mounted at `/models`. The selected main
path must be shard `00001`; Ember validates that the checksum set contains
every sibling through the declared `of-000NN` count. The set must also contain
the exact selected MTP, mmproj, and vocab-only text-model names. The artifact
manifest flattens these into the `mtp`, `vision_mmproj`, and `vision_vocab`
companion records. Startup validates the checksum-list digest and then all
listed files even when
`EMBER_VERIFY_EXISTING_SHA256=0`; that DeepSeek optimization never weakens the
Qwen boundary.

Copy [.env.example](../.env.example), fill the exact names and evidence digest,
and start the ordinary release service:

```dotenv
EMBER_DEPLOYMENT_MODE=qwen3.8-flash-next
EMBER_QWEN_MODEL=/models/Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-00001-of-000NN.gguf
EMBER_QWEN_SHA256SUMS=/models/SHA256SUMS
EMBER_QWEN_SHA256SUMS_SHA256=<64-lowercase-hex-from-release-evidence>
DFLASH_QWEN_MTP=/models/Qwen3.8-Flash-Next-MTP-ROCmI4-Strix-Halo.gguf
DFLASH_QWEN_MTP_DEPTH=<certified-1-through-4>
DFLASH_QWEN_VISION_MMPROJ=/models/Qwen3.8-Flash-Next-BF16-mmproj.gguf
DFLASH_QWEN_VISION_TEXT_MODEL=/models/Qwen3.8-Flash-Next-vocab-only.gguf
```

```bash
docker compose up -d
docker compose logs -f ember
```

The release image fixes `DFLASH_QWEN_VISION_PROVIDER` to its packaged dynamic
provider. The entrypoint also requires that shared object to be a readable
regular file before launching the server. This wires the runtime boundary.
Certification comes only from the exact `vision-certified.json` produced by
`qwen-gfx1151-vision.yml`; it covers a pinned image-text corpus twice (cold and
warm), a direct pinned-llama.cpp embedding differential with explicit float32
tolerances, and raw phase-separated RSS/HWM/GTT/UMA residency. Its GitHub
attestation is verified immediately and again at the publication boundary.
Require that evidence,
the ordinary gfx1151 runtime record, and
ordinary text/MTP certification before describing the deployment as a
certified multimodal release.

## Opt-in CPU/GPU/NPU deployment

The `release-xdna` image is a measured prototype, not the default release
backend. Its current Gen53 placement keeps the target model and authoritative
verification on the GPU, runs the resident DSpark projection/shared-expert
pipeline on XDNA2, and uses AVX-512 CPU code for DSpark routing and routed
ROCMFP4 experts. Two resident sessions are required to overlap one session's
NPU proposal with another session's GPU verification.

Confirm the host path before building:

```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
test -r /dev/accel/accel0
```

Build the release image. It contains only the serving plugin and the five
resident overlay artifacts; validators stay in the `dev-xdna` image:

```bash
docker build --network host --target release-xdna -f docker/Dockerfile \
  -t ember:xdna-local .
docker build --network host --target dev-xdna -f docker/Dockerfile \
  -t ember:xdna-dev .
docker run --rm --entrypoint ember-xdna-dspark-overlay-validate \
  --device /dev/accel/accel0 \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v "$PWD/models:/models:ro" \
  ember:xdna-dev \
  /usr/local/share/ember/xdna2 /models/DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf
```

Then start the opt-in overlay:

```bash
docker compose -f compose.yaml -f compose.xdna.yaml up --build -d
```

The overlay enables the DSpark provider and `--batch-sessions 2`; provider
failure falls back to GPU DSpark. Set `DFLASH_DSPARK_XDNA_REQUIRED=1` only for
correctness/performance gates that must reject fallback. The slower XDNA
target-expert provider and its historical kernel generations have been removed;
their measurements remain in the research guide.

The XDNA overlay defaults to authoritative q=1 prefix verification and a
four-row exact shadow capture. The latter never changes authoritative sparse
prefill. The 100-prompt frozen corpus and 15-case agentic suite pass, including
a cold replay of the cases that exposed q-wide rollback differences. Keep the
overlay experimental because the representative serial corpus remains 25.7%
slower than target-only; the fixed high-acceptance two-session fixture is not a
sufficient release-throughput gate. See
[`xdna2-moe-prototype.md`](xdna2-moe-prototype.md) for the measurements.

## Reproducible model acquisition

The container supports exactly this published pair at Hugging Face revision
`9fe32d8d4a1abed16c84e2636b26950232869929`:

- `DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf`:
  91,547,243,200 bytes, SHA-256
  `a936e0a514385c8ae964c0f42263a4314a34fbc6efea9d9aced5320f320a3d54`.
- `DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf`:
  10,897,111,840 bytes, SHA-256
  `1a01c80eceae302bcc1d70836759ee97974d7983c5084ef43f6ef772a8970ae6`.

Downloads first land in a `.part` file and resume after interruption. Ember
checks available space before transfer, verifies the completed staging file,
and only then renames it to the final `.gguf` path. A digest mismatch never
starts the server.

Other DeepSeek artifacts are unsupported. Its artifact filenames, revision,
and digests are not configurable. By default Ember also re-hashes pre-existing
DeepSeek artifacts on every start. On a trusted, immutable model store, set
`EMBER_VERIFY_EXISTING_SHA256=0` to skip only those startup scans. Downloads
remain verified before promotion regardless of this setting. The DeepSeek
pair has its own license on its Hugging Face card and is not included in the
Ember image. Qwen candidates use the sealed, local-only procedure above and
are never selected by merely dropping another GGUF into `/models`.

## Runtime state and observability

- `GET /health` is the readiness/liveness probe.
- `GET /status` exposes queue, worker, batching, cache, tool-loop, progress,
  degenerate-output, visible-markup, and automatic-recovery state.
- `GET /v1/models` confirms the client-visible model identifier.
- Container logs go to stdout/stderr; use `docker compose logs` or the host's
  configured Docker log driver for retention.
- Ember sends no usage telemetry and does not contact a control plane. With
  automatic download enabled, startup contacts only the configured Hugging
  Face repository.

Prometheus metrics and OpenTelemetry traces are intentionally not claimed in
the initial release. Operators needing time-series data can poll `/status` and
record client-side latency, but should not treat that JSON schema as a stable
Prometheus contract.

## Network and security boundary

The server binds to loopback by default, and Compose uses host networking so
that default survives containerization. Ember has no built-in authentication.
Set `EMBER_HOST=0.0.0.0` only when a trusted gateway in another network
namespace must reach it, and require authentication, TLS, request-size/rate
limits, and network policy at that boundary.

The ROCm runtime requires direct device access, host IPC, and an unconfined
seccomp profile in the supported Compose configuration. These privileges make
the container a workload-isolation convenience, not a security sandbox. Keep
the host and image patched and do not run untrusted models or modified engine
code on a sensitive host.

## Updating and rollback

1. Record the current source revision and `.env` overrides.
2. Fetch the desired `vYEAR.MONTH.DAY` Ember tag, confirm that it matches
   `VERSION`, and review `CHANGELOG.md`.
3. Build with explicit `EMBER_VERSION` and `EMBER_VCS_REF` values when producing
   a named local artifact.
4. Run the GPU-free suite, then the differential validator described in the
   README on the target machine. XDNA candidates additionally require the
   packaged provider gate and trained two-session differential/throughput gate.
5. Rebuild, start, and run `scripts/smoke_test.sh --generate`.

The model and KV cache are host directories, so rolling back the image does not
remove them. KV snapshots are disposable acceleration state, not backups; stop
Ember before copying or removing the cache directory.

### Release verification checklist

For a production upgrade, keep the checks in this order so a fast smoke test
cannot hide a model, image, or hardware mismatch:

1. Confirm the tag and `VERSION` agree, then read the matching section of
   [`CHANGELOG.md`](../CHANGELOG.md).
2. Run `scripts/preflight.sh` and record its device, memory, disk, and Docker
   output.
3. Start the image and wait for `/health`; run
   `scripts/smoke_test.sh --generate`.
4. On the exclusive gfx1151 host, run the differential validator with the
   disposable KV directory below. For `--batch-sessions 2`, require both
   resident streams to match the serial reference.
5. If the candidate enables XDNA2, repeat the provider validator with
   `DFLASH_DSPARK_XDNA_REQUIRED=1`; otherwise a GPU fallback can make an
   apparently successful test meaningless.

The validator must use a prompt that is safe to regenerate and a disposable
cache. Run it from the ROCm build directory (or replace the binary path):

```bash
printf '%s\n' 'Return the first 32 positive integers, one per line.' \
  > /tmp/ember-validation-prompt.txt
./build-rocm/ember-dflash \
  -m /models/DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf \
  --kv-cache-dir "$(mktemp -d /tmp/ember-validation-cache.XXXXXX)" \
  --validate-prompt /tmp/ember-validation-prompt.txt \
  --validate-tokens 32
```

Append `--batch-sessions 2` when certifying resident batching. The command exits
nonzero on a greedy mismatch after snapshot restore, disk round-trip, or the
speculative path; keep its complete output with the release record.

The release workflow performs these checks against immutable image and model
digests. A local dashboard bundle is useful for comparison, but its
`certified` field must be true before calling a result certified.

## Common failures

| Symptom | Meaning and action |
|---|---|
| `/dev/kfd is unavailable` | Install/fix the host AMD device stack and pass the device to Docker. |
| `no gfx1151 agent was found` | This image is running on unsupported hardware or ROCm cannot enumerate it. |
| `/dev/accel/accel0` is missing | The optional NPU path lacks a loaded/compatible `amdxdna` host driver, firmware, permissions, or device mount. |
| XRT reports mapping/`ENOSPC` failures | Confirm IOMMU translation is enabled and memlock is unlimited; `amd_iommu=off` is unsupported. |
| `insufficient free space` | Free space in `EMBER_MODELS_DIR` or move it to a larger local filesystem. |
| `model SHA-256 mismatch` | Remove the named final or `.part` file and retry; do not disable verification to hide corruption. |
| `Qwen SHA256SUMS ...` | The opt-in Qwen bundle is incomplete, unsafe, or differs from its release evidence. Restore the exact bundle; do not regenerate a checksum list around changed files. |
| `Qwen vision provider is not a readable regular file` | Use the matching release image with its packaged provider; do not point a certified artifact set at an arbitrary shared object. |
| container remains `starting` | Model load is still in progress; inspect logs. If the server exited, Compose will show the startup error. |
| clients cannot connect | Confirm `/health`, host/port overrides, and routing. The default `127.0.0.1` bind is intentionally unreachable from another network namespace; use `EMBER_HOST=0.0.0.0` only behind a trusted gateway. |
| agent behaves differently | Check the exact client settings in `client-compatibility.md`, especially base URL, model id, context, and tool mode. |

Include the output of `scripts/preflight.sh`, `docker compose logs ember`, the
source revision, model digest, kernel/ROCm versions, and reproduction request in
support reports. For XDNA runs also include the `amdxdna`/firmware/XRT versions,
IOMMU state, provider mode, and whether fallback was allowed. Redact prompts and
tool results that contain private data.
