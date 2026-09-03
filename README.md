# Ember

<p align="center">
  <img src="docs/assets/ember-logo.png" alt="Ember logo" width="240">
</p>

<p align="center">
  <strong>DeepSeek-V4-Flash inference for AMD Strix Halo</strong>
</p>

> [!IMPORTANT]
> **Ember is designed exclusively for AMD Strix Halo (`gfx1151`).** It is an
> intentionally focused engine for one APU platform and one model family,
> allowing its CPU, GPU, unified memory, and optional XDNA2 NPU path to be
> tuned together. It is not a general-purpose inference engine and does not
> support other GPU or NPU architectures.

Ember is a C inference server for DeepSeek-V4-Flash. It provides OpenAI Chat
Completions, OpenAI Responses, Anthropic Messages, and legacy Completions APIs
on one local endpoint.

Since 2026.9.3 the default model is **DeepSeek-V4-Flash-Vision-Exp**, so Ember
answers questions about images as well as text. [Sending an
image.](#send-an-image)

**23-40 tok/s decode** on a single Ryzen AI Max 395+. The latest
2026.8.24 release-candidate sweep measured **39.59 tok/s median** on a
structured workload (23.69 tok/s autoregressive baseline); prose and code are
slower, as shown below. These are engineering measurements from the pinned
model pair, not a promise for every prompt. The 194 ms warm-TTFT figure is
from the long-context cache run documented in the performance section.
[Details below.](#performance)

## Requirements

- Native x86_64 Linux on AMD Strix Halo (`gfx1151`)
- Approximately 128 GiB unified memory
- At least 100 GiB free disk space for the minimum preflight check; 120 GiB is
  recommended so a resumed download has room for staging and filesystem
  headroom
- Docker Engine with Docker Compose v2
- `/dev/kfd` and `/dev/dri` available to Docker

Docker Desktop, virtualized GPU passthrough, and other GPU architectures are not
supported. The normal release uses the GPU and CPU only. The opt-in XDNA2 image
additionally requires a working host `amdxdna` driver, firmware, IOMMU
translation, and `/dev/accel/accel0`.

## Start Ember

From the repository root:

```bash
scripts/preflight.sh && docker compose up -d
```

The first start pulls the [current Ember release](https://github.com/otheru-ai/ember/releases/latest)
image pinned in [`compose.yaml`](compose.yaml), downloads the pinned
[DeepSeek-V4-Flash-Vision model, vision tower and DSpark drafter](https://huggingface.co/otheru/DeepSeek-V4-Flash-Vision-Strix-Halo-GGUF)
(about 96 GiB combined), verifies all three, and starts Ember at
`http://127.0.0.1:8080`. Interrupted downloads resume automatically.

To serve the older text-only model instead, point `EMBER_MODEL_REPO` and
`EMBER_MODEL_REVISION` at
[the 0731 release](https://huggingface.co/otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF).

Follow startup progress:

```bash
docker compose logs -f ember
```

Model loading can take several minutes. When the service is healthy, run:

```bash
scripts/smoke_test.sh --generate
```

Later starts need only:

```bash
docker compose up -d
```

To build the release image from source instead of pulling it:

```bash
docker compose -f compose.yaml -f compose.build.yaml up --build -d
```

Stop Ember with:

```bash
docker compose down
```

Downloaded models remain in `./models`; the persistent KV cache remains in
`./cache`.

## Send a request

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

## Send an image

Images go in an OpenAI-style content part, either as a `data:` URL or a plain
base64 payload:

```bash
IMG=$(base64 -w0 photo.png)
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"deepseek-v4-flash\",
    \"messages\": [{\"role\": \"user\", \"content\": [
      {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMG\"}},
      {\"type\": \"text\", \"text\": \"What is in this image?\"}
    ]}],
    \"stream\": false
  }"
```

> [!IMPORTANT]
> **PNG only.** Ember decodes non-interlaced RGB/RGBA8 PNG and refuses JPEG,
> WebP and GIF rather than handing request bytes to a permissive decoder.
> Convert before sending.

An image request runs one full-model prefill. Speculative decode and resident
batching are declined for it rather than silently degraded, so an image request
is slower than a text request of the same length.

Useful endpoints:

- `GET /health`
- `GET /status`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/messages`
- `POST /v1/completions`

Ember has no built-in authentication and listens on loopback by default. Use an
authenticating reverse proxy before exposing it to another host. Set
`EMBER_HOST=0.0.0.0` (or pass `--host 0.0.0.0` to the binary) when a trusted
container or Kubernetes gateway must reach the API.

## Performance

[**Performance dashboard**](https://otheru-ai.github.io/ember/perf/) — every
measurement below, browsable by workload, release, and context depth.

One Ryzen AI Max 395+ (`gfx1151`, 128 GB), DeepSeek-V4-Flash-0731 at
ROCMFPX 2.5 bpw, 85.3 GiB resident. Each row is one request: a prompt of the
given depth, 256 tokens generated, greedy. Depth is the prompt size asked
for; prompt tokens is what it actually tokenised to, and every rate is
computed from that. `tok/s total` counts prompt plus generated tokens over
the whole request, so it folds prefill and decode together.

| Depth | Prompt tokens | tok/s out | Prefill tok/s | tok/s total | TTFT ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 43 | 39.19 | 71.0 | 41.9 | 606 |
| 1k | 862 | 39.43 | 281.0 | 116.9 | 3,068 |
| 4k | 3,925 | 38.00 | 342.0 | 229.6 | 11,477 |
| 16k | 18,553 | 32.62 | 299.2 | 269.3 | 62,009 |
| 32k | 38,059 | 24.79 | 271.0 | 254.1 | 140,439 |
| 64k | 77,068 | 16.58 | 223.5 | 214.6 | 344,823 |
| 96k | 116,077 | 14.86 | 184.1 | 179.6 | 630,511 |

Decode depends on how predictable the output is, because speculative decoding
pays in proportion to how well the drafter guesses the continuation:

| Output | tok/s |
| --- | ---: |
| Repetitive or structured (counting, JSON, lists) | 40 |
| Code, structured factual answers | 30 |
| Prose, essays, creative writing | 23 |

23 tok/s is also the rate with speculation off, so it is the floor rather than a
bad case.

TTFT above is cold. The prefix cache restores a conversation after the first
turn, and that is the figure agent and chat loops actually see: a 6,063-token
prompt whose first 6,053 tokens are cached prefills in **194 ms** instead of
17,693 ms.

Full methodology and the per-kernel roofline position are in
[docs/performance.md](docs/performance.md).

### Latest release-candidate sweep

The 2026.8.24 bundle was measured on 2026-08-23 with the same Ryzen AI Max+ 395
(`gfx1151`, 125 GiB installed unified memory) and 85.3 GiB target model. These
are three-sample medians
from the release-candidate harness; they are a current engineering reference,
not a replacement for the dedicated certification gate:

| Workload | Speculative tok/s | Autoregressive tok/s | Speedup |
| --- | ---: | ---: | ---: |
| Structured alphabet | 40.75 | 23.69 | 1.72× |
| Counting | 40.60 | 23.42 | 1.734× |
| Code | 28.92 | 23.86 | 1.212× |
| Repetitive text | 39.97 | 23.70 | 1.686× |

Across the three-run 256-token throughput sweep, the median was 39.59 tok/s
with a 98.1% median draft acceptance rate.

The corresponding prefill medians were 216.1 tok/s at 128 tokens, 412.8 at
2,048, 352.5 at 8,192, and 324.3 at 16,384. See the [performance
dashboard](https://otheru-ai.github.io/ember/perf/) for the complete bundle and
provenance. The dashboard marks this bundle `certified: false`: its repeatable
workload harness is separate from hardware certification, which additionally
runs the differential validator, resident-session checks, and a live request
against immutable image and model digests.

## Configure

The defaults require no configuration. To change the port or host directories:

```bash
cp .env.example .env
```

Edit `.env`, then restart with `docker compose up -d`. Common settings are:

```dotenv
EMBER_HOST=127.0.0.1
EMBER_PORT=8080
EMBER_MODELS_DIR=./models
EMBER_CACHE_DIR=./cache
```

The default `deepseek-v4-flash` deployment uses the pinned quant, the vision
tower and the drafter. Ember always verifies downloads before making them
runnable, and a missing vision tower stops startup rather than falling back to
the text path, so an image request can never be answered by a server that
cannot see it.

Two variables move the deployment to a different published model, both of which
must be set together:

```dotenv
EMBER_MODEL_REPO=otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF
EMBER_MODEL_REVISION=<immutable HuggingFace commit>
```

The revision is a commit, not a branch: a branch name would let the model change
between container restarts. Ember refuses to start with an unpinned revision
unless `EMBER_AUTO_DOWNLOAD=0` and the artifacts are supplied by hand. The first launch of an existing DeepSeek artifact hashes
it and stores an identity-bound integrity record under the persistent cache
mount. Later launches skip the content scan while device, inode, size, mtime,
and ctime still match. A changed or replaced file is hashed again. Operators
using a trusted, immutable model store may set `EMBER_VERIFY_EXISTING_SHA256=0`
to skip even the first verification.

An explicit, local-artifact-only Qwen deployment boundary is also available for
a completed Qwen3.8-Flash-Next candidate. It does not silently reuse the
DeepSeek drafter or guess an MTP depth, and it does not download unpublished
candidate files. Place the complete candidate package in the model directory,
then set:

```dotenv
EMBER_DEPLOYMENT_MODE=qwen3.8-flash-next
EMBER_QWEN_MODEL=/models/Qwen3.8-Flash-Next-Heretic-ROCmI4-Strix-Halo-00001-of-000NN.gguf
EMBER_QWEN_SHA256SUMS=/models/SHA256SUMS
EMBER_QWEN_SHA256SUMS_SHA256=<digest-from-the-certified-release-evidence>
DFLASH_QWEN_MTP=/models/Qwen3.8-Flash-Next-MTP-ROCmI4-Strix-Halo.gguf
DFLASH_QWEN_MTP_DEPTH=<certified-1-through-4>
DFLASH_QWEN_VISION_MMPROJ=/models/Qwen3.8-Flash-Next-BF16-mmproj.gguf
DFLASH_QWEN_VISION_TEXT_MODEL=/models/Qwen3.8-Flash-Next-vocab-only.gguf
```

Pass shard `00001`; the engine discovers every ordered sibling. Startup always
verifies the small checksum-list digest and requires the list to cover all main
shards plus the selected MTP, BF16 mmproj, and zero-tensor vocab-only vision
text model. Files named by the sealed list use the same identity-bound cache,
regardless of `EMBER_VERIFY_EXISTING_SHA256`: the first launch hashes them and
later launches hash only files whose identity changed. The release artifact
manifest exposes those as three flattened
companion records (`mtp`, `vision_mmproj`, and `vision_vocab`). The release
image supplies the pinned `DFLASH_QWEN_VISION_PROVIDER` shared object. Until a candidate has its
real-weight text/vision differential and gfx1151 release evidence, this mode is
an engineering deployment boundary rather than a multimodal certification
claim. See [.env.example](.env.example) for every container setting and
[the operations guide](docs/operations.md) for the fail-closed startup checks.

## Coding agents

Ember is tested with Claude Code, Codex CLI, OpenCode, pi, and OMP. Their exact
provider configurations are in
[docs/client-compatibility.md](docs/client-compatibility.md).

The default base URL is `http://127.0.0.1:8080/v1` and the model ID is
`deepseek-v4-flash`.

## Develop and test

Open the full ROCm development container:

```bash
docker compose --profile dev run --rm --build ember-dev
```

Build and run the GPU-free test suite on any normal C/C++ host:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Build the real ROCm backend from the repository root with:

```bash
docker build --network host --target dev -f docker/Dockerfile \
  -t ember-rocm:10.0-dev .
scripts/build.sh
```

The image is pinned to ROCm 10.0.0 and includes ROCm Compute Profiler 3.8.0;
see [the ROCm 10 migration note](docs/rocm-10.md) before comparing new
roofline output with older profiling bundles.

### CPU/GPU/NPU prototype

The opt-in `release-xdna` image packages the XRT/IRON provider used by Ember's
measured heterogeneous decode prototype. Its current division of work is:

| Processor | Role |
|---|---|
| GPU | Target model, DSpark main projection, authoritative q=1 prefix verification |
| NPU | Asynchronous resident DSpark projection and shared-expert pipeline |
| CPU | DSpark routing, AVX-512 ROCMFP4 routed experts, orchestration |

Two resident sessions overlap NPU proposal work for one request with GPU target
verification for another. The best fixed-fixture run measured a 1.484x
aggregate throughput speedup. The current correctness-first path subsequently
matched a fresh target-only reference on all 100 frozen prompts and passed all
15 agentic cases, including a cold-start replay of the nine cases that exposed
an earlier rollback bug. On that representative serial corpus it remained
25.7% slower than target-only, so the path is still experimental and opt-in;
it is not the default release backend.

It requires a compatible host `amdxdna` driver and firmware, enabled IOMMU,
`/dev/accel/accel0`, and render-group access. Build and start it with:

```bash
docker compose -f compose.yaml -f compose.xdna.yaml up --build -d
```

The overlay selects two resident sessions and falls back to ordinary GPU
DSpark if the NPU provider cannot initialize. Read the
[XDNA2 heterogeneous inference guide](docs/xdna2-moe-prototype.md) before
validation or benchmarking.

## Documentation

- [Architecture and backend boundaries](ARCHITECTURE.md)
- [Release notes](CHANGELOG.md)
- [CI and hardware certification](docs/ci.md)
- [Operations and troubleshooting](docs/operations.md)
- [Continuous batching](docs/continuous-batching.md)
- [Measured performance: TTFT, decode, roofline](docs/performance.md)
- [Quantisation quality and release gates](docs/quant-quality-reports.md)
- [Experimental CPU/GPU/XDNA2 inference](docs/xdna2-moe-prototype.md)
- [Lessons from upstream engines](docs/upstream-lessons.md)
- [Client configuration](docs/client-compatibility.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Vendor provenance](engine/VENDOR.md)

The contributor-facing contract is in [`AGENTS.md`](AGENTS.md). It records the
source-list invariants, GPU validation protocol, and architectural rules that
are easy to break while changing the server or vendored engine. Keep it and
[`CLAUDE.md`](CLAUDE.md) aligned when build or test conventions change.

Ember is licensed under the [MIT License](LICENSE). Model weights are distributed
separately under the license on their Hugging Face model card.
