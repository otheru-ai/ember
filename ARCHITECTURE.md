# Ember architecture

Ember is a model-specific inference server for **DeepSeek-V4-Flash on AMD
Strix Halo (`gfx1151`)**. It owns the HTTP and agent-facing behavior, while a
vendored engine owns the model forward pass and the hardware-specific kernels.

The shortest accurate description is:

> Ember is a C protocol server around a stable backend ABI, with a C++ bridge
> to a maintained fork of lucebox's DeepSeek and ROCm engine.

This boundary is the central architectural decision. Ember is not a general
GGUF runner, and it is not a clean-room kernel implementation. The server was
rewritten so request handling, streaming, tool use, caching policy, and errors
are explicit and testable. The tuned tokenizer, model implementation, and GPU
kernels were retained because they are model-coupled and performance-critical.
An opt-in provider below that boundary can also pipeline DSpark work across the
Strix Halo CPU and XDNA2 NPU without exposing either runtime to the C server.

For supported APIs, commands, flags, and measured performance, start with
[`README.md`](README.md). This document explains how the pieces fit together
and where changes belong.

## System at a glance

```text
OpenAI / Anthropic / agent clients
                  |
                  | HTTP/1.1 + JSON/SSE
                  v
+------------------------------------------------------------+
| HTTP and protocol layer (fresh C)                          |
| src/server/: routing, validation, adapters, wire output    |
+------------------------------------------------------------+
                  |
                  | ember_chat_request
                  v
+------------------------------------------------------------+
| Request orchestration and model semantics (fresh C)        |
| main.c + src/model/: DSML, tools, compaction, KV policy    |
+------------------------------------------------------------+
                  |
                  | stable C ABI
                  v
+------------------------------------------------------------+
| src/backend/ember_backend.h                                |
| tokenizer, generation, snapshots, disk cache, metrics      |
+--------------------------+---------------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
  backend_stub.c                 backend_dflash.cc
  tests and local builds         ROCm C++ bridge
                                         |
                                         v
                              +-------------------------+
                              | Vendored engine         |
                              | DeepSeek, ROCMFP, MoE,  |
                              | attention, DSpark       |
                              +------------+------------+
                                           |
                               optional DSpark provider ABI
                                           |
                              +------------v------------+
                              | providers/xdna2/        |
                              | XRT/IRON NPU + AVX-512  |
                              | CPU draft pipeline      |
                              +-------------------------+
```

The major ownership boundaries are:

| Area | Location | Ownership |
|---|---|---|
| HTTP, API adapters, streaming, orchestration | `src/server/` | Ember code, written in C |
| Prompt semantics, tool validation, replay, cache policy | `src/model/` | Ember code, written in C |
| Model-facing contract | `src/backend/ember_backend.h` | Stable Ember C ABI |
| GPU bridge | `src/backend/backend_dflash.cc` | Ember-owned C++ adapter |
| DeepSeek implementation and ROCm kernels | `engine/` | Maintained vendored fork; see `engine/VENDOR.md` |
| Optional CPU/NPU provider and AIE kernels | `providers/xdna2/` | Ember prototype with pinned upstream provenance |
| Constrained-decoding runtime | `vendor/xgrammar/` | Vendored dependency; see its `VENDOR.md` |

## Why this boundary exists

The model tokenizer and forward pass are not interchangeable utilities. The
DeepSeek chat format depends on a byte-exact `joyai-llm` pre-tokenizer variant,
and the runtime depends on gfx1151-specific work across attention, the
256-expert MoE, hyper-connections, ROCMFP quantization, snapshotting, and
speculative decoding. Reimplementing those pieces would discard the reason the
engine is fast and risk silently changing token identity.

The server-facing behavior has different requirements: protocol correctness,
bounded resource use, tool safety, cancellation, replay, and understandable
errors. Those concerns are easier to reason about in a small C layer that does
not expose ggml or HIP types.

`ember_backend.h` separates the two. The server can be exercised against a
deterministic stub on any machine, while the deployed binary uses the same
calls through an `extern "C"` bridge to the real engine.

## Request lifecycle

All generation endpoints become one protocol-neutral request before reaching
the model. The main pipeline lives in `run_chat()` in `src/server/main.c`.

```text
HTTP request
  -> route and parse protocol JSON
  -> normalize to ember_chat_request
  -> attach exact tool replay or restore a continuation frontier
  -> render DSML and encode, preserving token splices
  -> optionally compact, then re-render and re-encode
  -> enforce the context limit
  -> resolve sampling, reasoning budget, stops, and tool grammar
  -> choose the longest safe KV prefix and reserve a snapshot target
  -> generate through ember_backend_generate()
  -> split reasoning/text/tools and validate executable tool calls
  -> emit native buffered JSON or SSE events
  -> commit successful cache/replay state and release the generation
```

### 1. Route and normalize

`http.c` parses a deliberately small HTTP/1.1 surface. `main.c` routes OpenAI
Chat Completions, OpenAI Responses, legacy Completions, and Anthropic Messages.
`chat_api.c` and `api_adapters.c` translate those request shapes into
`ember_chat_request`.

The internal pipeline does not branch on raw client JSON. Protocol differences
return at the output boundary, where Chat, Responses, Completions, and Messages
receive their native envelopes and streaming event taxonomies.

### 2. Reconstruct the authoritative prompt

`chat_template.c` renders DeepSeek's DSML chat format. If earlier assistant tool
calls are known, `tool_memory.c` substitutes only their exact sampled tool-call
blocks. `encode_with_splices()` then inserts the stored token IDs instead of
retokenizing special DSML markers.

Continuation-only requests take a stricter path. `continuation.c` binds the
complete call-ID set, API family, sampled token frontier, visible text, and tool
schema. A missing or mismatched binding fails rather than inventing history.

### 3. Compact before rejecting

When `--auto-compact` is enabled and the prompt approaches the context ceiling,
`compaction.c` asks the model for durable task state and rebuilds the request as
leading system/developer messages, a bounded summary, and a recent verbatim
tail.

Compaction intentionally runs before the context-length guard. A successful
compaction returns messages, not token IDs, so the request is rendered and
splice-encoded again. The client-visible `usage.prompt_tokens` remains the size
of the request the client sent; compacted and internally generated tokens are
reported separately.

Compaction generations never create reusable KV entries. This prevents the
server's private summary prompt from becoming a prefix restored by an unrelated
request.

### 4. Plan generation and cache reuse

Model-card defaults and explicit request values are resolved independently.
The server also prepares stop strings, reasoning force-close tokens, optional
DRY sampling, structural greedy steering, and a request-specific tool grammar.

The prefix cache selects the longest safe in-memory or disk checkpoint, pins
the restore source, and reserves a different destination when the new prompt
has a reusable boundary. The backend evaluates only the uncached suffix.

### 5. Generate and project onto the client protocol

`ember_backend_generate()` calls back once per token and periodically during
prefill. Callbacks detect disconnects, send streaming deltas or keepalives, and
can cancel work when the client is gone.

Buffered responses collect the complete model turn before constructing a
protocol response. Streaming responses use the same accumulated output as the
source of truth; protocol sinks translate safe deltas into each API's event
format.

### 6. Commit only proven state

Logical cache entries are committed only after the backend confirms that the
corresponding snapshot was saved. Exact tool blocks and continuation bindings
are retained only for successful, validated generations. Completion and error
paths release resident generation state, snapshot pins, request buffers, and
serialization locks before returning.

## Concurrency and ownership

HTTP connections and model generations have different lifetimes, so Ember does
not run the model directly on connection threads.

`http.c` creates one thread per accepted connection. A generation request is
placed on a bounded queue, and the connection thread waits for its job. The
queue holds at most eight pending jobs; excess work receives HTTP 503 instead
of consuming memory without bound. Foreground requests take priority over jobs
marked `ember_background:true`, while a maximum wait prevents background work
from starving forever.

Read-only control endpoints bypass the generation queue:

- `/health`
- `/status`
- `/v1/models`
- `/v1/models/{id}`

They remain responsive during a long prefill or decode.

Ember has two execution modes:

| Mode | Server dispatch | Backend ownership |
|---|---|---|
| `--batch-sessions 1` | One persistent generation worker; requests are serialized | One mutable model frontier and thread-local graph caches |
| `--batch-sessions 2..64` | A bounded worker pool admits overlapping requests | The engine coordinator owns isolated resident sessions and serializes engine submissions |

The persistent legacy worker is required because DeepSeek compute graphs and
scratch arenas are thread-local. Recreating a short-lived thread for each
request repeatedly rebuilt and abandoned a roughly 918 MB prefill arena.
Backend teardown and idle graph reclamation must therefore occur on the owning
thread. `EMBER_IDLE_RECLAIM_SECS` releases graph arenas after a quiet period
without unloading model weights.

In resident mode, each admitted session owns its KV, HC/compressor state,
logits, sampler/RNG history, pending tokens, callbacks, and timing. Model
weights and the coordinator remain shared. The scheduler interleaves bounded
prefill quanta and decode-ready sessions, but the current DeepSeek decode batch
uses a correctness-first serial row fallback. Ordinary GPU DSpark remains on
the monolithic path; the opt-in XDNA2 provider has a session-isolated resident
proposal/verifier path. See
[`docs/continuous-batching.md`](docs/continuous-batching.md).

Shared server bookkeeping—prefix reservations, replay memory, continuation
bindings, and telemetry—is protected by `state_lock`. The legacy forward pass
is additionally protected by `gen_lock`; resident sessions bypass that lock
because their mutable engine state is isolated below the ABI.

## Streaming and protocol safety

`sse.c` uses a **buffer-and-resplit** design. The caller retains all generated
bytes and asks the splitter to rescan the full output after each token. The
splitter advances an emission watermark only over bytes known to be safe.

This is intentional. An incremental fixed window can lose the opening half of
a split `</think>` marker, DSML tool marker, UTF-8 codepoint, or client stop
sequence. Retaining the full buffer makes every incomplete boundary
re-discoverable regardless of how the tokenizer divided it.

The splitter is protocol-neutral. OpenAI Chat uses its built-in emitter;
Responses and Anthropic bind a sink that produces their native events. The same
reasoning/text/tool interpretation therefore drives both streaming deltas and
terminal objects.

Tool output crosses an executable-validation boundary before it is exposed as
a structured call. The pipeline checks framing, JSON syntax and duplicate
keys, advertised names, `tool_choice`, parallel-call policy, object arguments,
and recursive JSON Schema constraints. XGrammar can mask invalid structural
tokens while the model is inside a tool-call block; the post-generation
validator remains authoritative.

Tool-bearing streams withhold the candidate block until it validates, so a
client never receives a partial executable call. A malformed block is not
emitted as executable; by default the stream finishes normally with the
protocol's non-executable fallback, matching the server's ds4 compatibility
boundary. `EMBER_STREAM_TOOL_ERROR=1` opts into a typed terminal error instead.
Atomic requests may make one hidden, model-visible correction attempt before
returning a typed failure.

## State and caching

Ember is request-stateless at the HTTP boundary—the client normally resends its
conversation—but it keeps acceleration and exact-replay state inside the
process and, when configured, on disk.

| State | Purpose | Persistence |
|---|---|---|
| Model weights and tokenizer | Forward pass and byte-exact tokenization | Loaded for process lifetime |
| Compute graphs and scratch arenas | Avoid rebuilding shapes repeatedly | In memory; reclaimable while idle |
| Resident session state | Isolate concurrent KV, logits, sampler, and callbacks | In memory until explicit release |
| `ember_kv_cache` | Map token prefixes to backend snapshot slots | Logical index in memory |
| Backend snapshots | Hold model KV/HC state for reusable prefixes | In memory; selected checkpoints may be stored on disk |
| `ember_tool_memory` | Preserve exact sampled DSML bytes and token IDs | Bounded memory with optional model-scoped disk storage |
| Continuation store | Bind call IDs to an authoritative sampled frontier | Bounded memory with optional model-scoped disk storage |

### Prefix snapshots

`kv_cache.c` contains policy, not tensors. It finds the longest stored token
prefix, chooses a future checkpoint at a completed turn boundary or cold prompt
anchor, reserves slots, pins active restore sources, and applies LRU eviction.
The vendored disk cache supplies snapshot serialization; Ember owns checkpoint
timing and logical eviction policy.

The backend's `snapshot_pos()` is the authoritative number of tokens covered by
a slot. It is not necessarily the number of emitted tokens: ordinary decode
writes a token's KV row at the beginning of the next step, and speculative
decode can run ahead. Constructing a key from emitted length would claim state
the snapshot cannot restore.

The final backend slot is reserved for disk-restore staging, so configurable
in-memory prefix slots never consume it. A reserved logical slot is committed
only when the backend reports a successful snapshot save.

### Exact tool replay and continuation

DSML markers are special tokens and may not survive a
detokenize→retokenize cycle. Ember therefore stores the exact bytes and token
IDs for each sampled tool-call block. Only that protocol block is replayed;
reasoning and visible assistant text are rendered canonically.

For ID-only continuations, the complete sampled frontier is bound to the API
family and full call-ID set. Disk records are scoped by model/cache identity,
and dynamically sized ID arrays avoid an artificial parallel-call ceiling.
These constraints prevent a tool ID from replaying hidden or unrelated model
history.

## The backend ABI

`src/backend/ember_backend.h` is the only model-facing contract visible to the
C server. Its operations fall into five groups:

1. Model lifecycle and introspection.
2. Model-coupled encode and token-to-byte decode.
3. Generation with callbacks, sampling policy, constrained decoding, and KV
   restore/snapshot slots.
4. Disk snapshot and differential-validation operations.
5. Resident batching statistics, generation release, and idle graph reclaim.

There are two implementations:

- `backend_stub.c` is deterministic and GPU-free. It drives the real HTTP,
  prompt, tool, cache, and streaming pipeline in tests.
- `backend_dflash.cc` translates the C structures into the vendored C++ engine
  and converts its results back without leaking C++ types across the seam.

An ABI change is incomplete until both implementations and their tests are
updated. A new server source file must also be added to both explicit source
lists in `CMakeLists.txt`; the real executable cannot link `ember_core` because
that would collide the stub and real backend symbols.

## The vendored engine

`engine/` is a maintained source snapshot, not an opaque binary. It contains:

- the ggml fork and ROCm backends;
- ROCMFP quantized matrix kernels;
- DeepSeek-V4 attention, MoE, hyper-connection, and model-loading code;
- DSpark speculative decoding and verification;
- in-memory and disk snapshot machinery;
- resident scheduling and coordination.

Its upstream origin and pinned commits are recorded in
[`engine/VENDOR.md`](engine/VENDOR.md). Engine changes are allowed, but they are
fork divergence: port upstream work against the pinned revision, preserve
license/provenance records, and identify local changes clearly.

HIP graph replay is deliberately disabled. On gfx1151, measured capture churn
made it slower than the normal path. The benchmark and gate are documented next
to the option in `engine/CMakeLists.txt`; it should not be enabled until a stable
graph key is demonstrated by measurement.

## Heterogeneous CPU/GPU/NPU execution

The normal release remains the authoritative GPU/CPU implementation. The
opt-in `release-xdna` image adds a versioned DSpark provider seam below the
backend ABI; HTTP, model semantics, snapshots, and the target model do not know
about XRT.

The current Gen53 resident placement assigns work by measured strength:

| Processor | Owned work |
|---|---|
| gfx1151 GPU | Target prefill/decode, attention and target MoE, DSpark main projection, authoritative q=1 prefix verification |
| XDNA2 NPU | Resident Q8 DSpark projection and shared-expert AIE runlists |
| Zen 5 CPU | Draft routing, AVX-512 ROCMFP4 routed experts, accumulation, XRT/HIP ordering |

Each eligible resident session owns an asynchronous proposal job and captured
target-feature window. Sparse prefill remains on the target-only graph; an
isolated spare cache rebuilds four exact q=1 support rows, so capture cannot
perturb target logits or KV. The coordinator submits NPU work for one session
while the GPU verifies another, then commits only the prefix accepted by the
ordinary fused q=1 target graph. A q-wide prefilter was rejected because it
perturbed cold-start output even after rollback. A provider initialization or
execution failure falls back to ordinary GPU DSpark unless
`DFLASH_DSPARK_XDNA_REQUIRED=1` makes the validation boundary fail closed.

Strix Halo's unified physical memory does not make HIP and XRT ownership
implicit. The CPU orders completion and synchronization; provider weights and
buffers still cross explicit runtime ownership boundaries. Direct HIP/XRT
dma-buf interoperability has been validated, but it does not create autonomous
GPU-to-NPU dispatch—the CPU remains the command and fence authority.

This path has a measured high-acceptance two-session throughput win. Its direct
q=1 commit path also matched a fresh target-only reference on the 100-prompt
frozen corpus and passed the 15-case agentic suite, including cold-start
coverage. The same representative serial corpus measured 14.876 tok/s against
20.031 tok/s target-only, so the optional overlay still fails the release
throughput gate. The complete measurements, rejected placements, and promotion
gates are in
[`docs/xdna2-moe-prototype.md`](docs/xdna2-moe-prototype.md).

## Deployment architecture

The Dockerfile has two intentionally different outputs:

```text
ROCm toolchain image
        |
        +-- dev target
        |     full compiler/SDK + source + symbols + build tree
        |
        +-- release target
              Ubuntu base + stripped server + crash shim
              + exact recursive ROCm runtime dependency closure
              + gfx1151 rocBLAS data + download utilities

        +-- dev-xdna / release-xdna targets (opt-in)
              release contents + pinned XRT/XDNA userspace
              + IRON AIE artifacts + heterogeneous provider
```

The model is not included in either image. Compose mounts `./models` and
`./cache`, downloads the pinned GGUF on first use, verifies its SHA-256, and
then starts the release image. Host networking preserves Ember's loopback
default; `--host`/`EMBER_HOST` can explicitly select another IPv4 bind address.
Direct access to `/dev/kfd` and `/dev/dri`, host IPC, and an unconfined seccomp
profile are required by the supported ROCm deployment; the container is
therefore a packaging boundary, not a security sandbox.

The XDNA overlay additionally passes `/dev/accel/accel0`, locks accelerator
buffers in memory, selects two resident sessions, and enables the DSpark
provider. It packages XRT userspace and AIE artifacts only. The host kernel's
`amdxdna` module, NPU firmware, IOMMU configuration, and device permissions are
outside the image and must already be correct.

`libsegvtrace.so` is preloaded in the release image to print a symbolized
backtrace on fatal signals. This is more practical than a core dump for a
process whose resident model state approaches 100 GB.

## Failure and security boundaries

The architecture fails closed where ambiguity could execute the wrong action
and degrades gracefully where model output is merely unusable:

- malformed HTTP or protocol requests return native client errors;
- the request body is capped at 64 MB and sockets have bounded I/O waits;
- a full generation queue returns 503;
- prompts are compacted before the hard context check, then rejected honestly
  if they still do not fit;
- tool calls are never emitted as executable structures before validation;
- unsupported strict-schema keywords fail closed;
- failed snapshot writes never create logical cache hits;
- continuation IDs must match a model-scoped authoritative frontier;
- client disconnects cancel prefill/decode through callbacks;
- the server binds to loopback by default and provides no built-in
  authentication; selecting a wildcard address is an explicit operator action.

Tool-loop detection is observability by default, not a ceiling. Ember derives
request-local diagnostics from repeated call signatures, identical call/result
rounds, and a progress lease keyed by novel tool-result effects. It reports
them in response metadata, logs, and `/status`; non-progress output,
degeneracy, and visible tool-markup leaks are recorded separately. The optional
`--auto-answer-after-loop` policy is deliberately off by default. When enabled,
it suppresses optional tools for one turn and adds a private recovery
instruction, but never overrides a client-required tool choice or creates
cross-request state.

## Testing the architecture

The stub backend is what makes most architectural behavior testable without a
GPU. The normal CMake/CTest build covers protocol adapters, HTTP, SSE chunk
boundaries, prompt rendering, tool parsing/schema/grammar/replay, continuation,
compaction, model cards, KV policy, scheduling, and live server behavior.

Engine scheduler and coordinator tests compile selected C++ sources directly
on the host. The ROCm build then verifies that the bridge and vendored engine
compile together. On target hardware, the differential validator compares
fresh autoregressive output with in-memory restore, disk round-trip, DSpark,
and optional resident-session paths.

The XDNA provider has GPU-free ABI, packing, routing, queue-lifetime, and SIMD
tests plus packaged hardware validators for AIE kernels and HIP/XRT
interoperability. A release claim still requires the trained-model two-session
differential and throughput gates on a host with translated IOMMU domains.

This division is deliberate:

- behavior above the ABI should be reproducible in ordinary CI;
- model and GPU correctness must be proven on gfx1151;
- NPU correctness and overlap must be proven on the pinned XRT/firmware/driver
  tuple with IOMMU enabled;
- performance claims require measurement on the supported model and hardware.

## Where a change belongs

| Change | Primary location |
|---|---|
| Add or adjust an HTTP endpoint | `src/server/http.*`, routing in `main.c` |
| Change a client protocol shape | `chat_api.*` or `api_adapters.*` |
| Change streaming split semantics | `sse.*` and its chunk-boundary tests |
| Change DSML prompt rendering | `src/model/chat_template.*` |
| Change tool parsing or validation | `tool_parser.*`, `tool_schema.*`, `tool_grammar.*` |
| Change continuation or exact replay | `tool_memory.*`, `continuation.*`, splice logic in `main.c` |
| Change prefix selection or eviction | `kv_cache.*` and orchestration in `main.c` |
| Add a model/backend capability | `ember_backend.h`, both backend implementations, tests |
| Change kernels or DeepSeek execution | `engine/`, with `VENDOR.md` review |
| Change XDNA placement, provider ABI, or AIE kernels | `providers/xdna2/`, engine provider seam, provenance and hardware gates |
| Change runtime packaging | `docker/`, `compose.yaml`, release-script tests |

Before changing a subtle behavior, read the design block in the corresponding
header and the regression tests. Several choices that look unusual—full-buffer
stream splitting, compaction before the context guard, snapshot coverage from
the backend, exact-token tool splicing, persistent generation threads, and
disabled HIP graphs—are correctness or measured-performance decisions rather
than accidental complexity.

## Current scope and non-goals

Ember intentionally targets one model family and one integrated APU platform.
Current non-goals include:

- a general-purpose model registry or arbitrary GGUF compatibility;
- Metal, CUDA, multi-GPU, or tensor-parallel deployment;
- other NPU architectures or treating XDNA2 as a generic ggml backend;
- GLM model support;
- replacing the tuned HIP kernels solely to remove the vendor boundary;
- built-in authentication, TLS termination, or internet-facing tenancy;
- treating containers as isolation for untrusted model code;
- imposing a fixed number of agent tool rounds.

These constraints keep the supported path small enough to validate end to end.
New scope should be added behind explicit interfaces and tests, not by leaking
model or platform assumptions upward through the backend ABI.
