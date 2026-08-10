# Operations guide

This guide covers the supported deployment: one DeepSeek-V4-Flash
ROCMFPx model on a native Linux AMD Strix Halo (`gfx1151`) host. Ember is not a
general-purpose GGUF runner, and the release image is not expected to work on
other GPU architectures.

## Host prerequisites

- x86_64 Linux; Docker Desktop and virtualized GPU passthrough are unsupported.
- A ROCm-supported kernel/device stack exposing `/dev/kfd` and `/dev/dri`.
- AMD Strix Halo (`gfx1151`). The image is compiled specifically for this ISA.
- Approximately 128 GiB unified memory. The tested model is 85.3 GiB and uses
  about 89 GiB resident before context and operating-system overhead.
- At least 100 GiB free in the model directory for the 91,547,243,200-byte
  quant, 10,897,111,840-byte drafter, download staging, and filesystem
  headroom.
- Docker Engine with the Compose v2 plugin.

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

Other model artifacts are unsupported. The artifact filenames, revision, and
digests are not configurable, and verification cannot be disabled. The model
pair has its own license on its Hugging Face card and is not included in the
Ember image.

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

The server binds to loopback, and Compose uses host networking so that default
survives containerization. Ember has no built-in authentication. Do not change
the bind address or publish it to an untrusted network without an authenticating
reverse proxy, TLS, request-size/rate limits, and network policy.

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
   README on the target machine.
5. Rebuild, start, and run `scripts/smoke_test.sh --generate`.

The model and KV cache are host directories, so rolling back the image does not
remove them. KV snapshots are disposable acceleration state, not backups; stop
Ember before copying or removing the cache directory.

## Common failures

| Symptom | Meaning and action |
|---|---|
| `/dev/kfd is unavailable` | Install/fix the host AMD device stack and pass the device to Docker. |
| `no gfx1151 agent was found` | This image is running on unsupported hardware or ROCm cannot enumerate it. |
| `insufficient free space` | Free space in `EMBER_MODELS_DIR` or move it to a larger local filesystem. |
| `model SHA-256 mismatch` | Remove the named final or `.part` file and retry; do not disable verification to hide corruption. |
| container remains `starting` | Model load is still in progress; inspect logs. If the server exited, Compose will show the startup error. |
| clients cannot connect | Confirm `/health`, port overrides, and that the client uses `127.0.0.1`, not a container-only hostname. |
| agent behaves differently | Check the exact client settings in `client-compatibility.md`, especially base URL, model id, context, and tool mode. |

Include the output of `scripts/preflight.sh`, `docker compose logs ember`, the
source revision, model digest, kernel/ROCm versions, and reproduction request in
support reports. Redact prompts and tool results that contain private data.
