# AGENTS.md

Guidance for AI coding agents working in this repository. The reader is assumed
to know nothing about the project. Read this file fully before editing.

## Project overview

Ember is a from-scratch **C inference server for DeepSeek-V4-Flash on AMD Strix
Halo (gfx1151)**. It is
**ds4/Dwarfstar's server architecture rewritten clean in C, driving lucebox's
tuned HIP kernels** through a stable C ABI.

The load-bearing decision: the GPU kernels (attention, 256-expert MoE, ROCMFP
quant decode, DSpark speculative decode, KV snapshot/restore)
and the tokenizer (a `joyai-llm` pre-tokenizer variant that must be byte-exact)
are **reused** via a vendored engine — they are the entire performance advantage
and represent person-years of gfx1151-specific tuning. Everything above the
forward pass is **rewritten fresh in C** in this repo. This is a *server rewrite
with a kernel bridge*, not a kernel rewrite.

The published full-ROCMFP affine fp2 model (85.3 GiB, 2.58 bpw) meets
the Strix-Halo reference benchmarks (~248–253 tok/s sparse prefill, ~32 tok/s
decode with DSpark). See `README.md` for installation and first use, and
`ARCHITECTURE.md` for the layering rationale.

Primary documentation to consult, in order:

- `README.md` — prerequisites, container quick start, first request, and
  development commands.
- `ARCHITECTURE.md` — the layering and why the server was rewritten but the
  kernels reused.
- `docs/continuous-batching.md` — resident-session batching design.
- `docs/quant-quality-reports.md` — quant evaluation workflow and release gates.
- `CLAUDE.md` — a parallel guidance file with overlapping content; keep both
  files consistent when you change build/test/convention facts.

## Repository layout

```
src/server/    HTTP/1.1 (http.c), SSE streaming (sse.c), chat completions
               (chat_api.c), protocol adapters (api_adapters.c: OpenAI Chat,
               Responses, legacy Completions, Anthropic Messages), context
               compaction (compaction.c), background gating (background_gate.c),
               and main.c — the request-pipeline hub (run_chat).
src/model/     DSML chat template (chat_template.c), tool-call parsing
               (tool_parser.c, dsml_decode.c), exact-token tool replay
               (tool_memory.c), cross-restart continuation (continuation.c),
               KV prefix cache (kv_cache.c), GGUF metadata (gguf.c), model
               cards (model_card.c).
src/common/    buf.h (the universal growable buffer), json.c/json.h/json_util.h,
               utf8.h.
src/backend/   ember_backend.h — THE stable C ABI. Two implementations:
               backend_stub.c (deterministic, GPU-free; drives every test) and
               backend_dflash.cc (extern "C" shim over the vendored engine).
engine/        Vendored fork of lucebox: ggml fork with ROCMFP kernels
               (engine/ggml), the DeepSeek4 backend (engine/dflash/deepseek4),
               batching machinery (engine/dflash/common), HIP compat shims.
               Provenance pinned in engine/VENDOR.md.
test/          Plain C/C++ test binaries (one main() each) plus Python
               server-level and quant-pipeline tests.
scripts/       build.sh (ROCm container build), diagnostics, and the quant
               quality pipeline (*.py, stdlib-only).
share/         model_cards/ (per-model defaults sidecar JSON + _schema.json),
               quant_eval/ (eval fixtures).
reports/       Generated quant quality reports (Markdown/JSON/CSV/SVG).
docker/        Multi-stage Dockerfile: full-ROCm `dev` toolchain and minimal
               dependency-closure `release` image (Compose default).
tools/         segvtrace.c — crash-backtrace shim LD_PRELOAD'd in production.
docs/          Design/audit documents listed above.
```

There are no `pyproject.toml`/`package.json`/`Cargo.toml` files: the build is
pure CMake, and the Python scripts use only the standard library.

## Build and test

Two build configurations. **Almost all work happens in the first one.**

```bash
# GPU-free: server + stub backend + full test gauntlet. Builds on any host.
cmake -S . -B build && cmake --build build && ctest --test-dir build

# Real backend (ROCm/HIP; MUST run in the container — no HIP toolchain on host)
docker build --target dev -f docker/Dockerfile \
  -t ember-rocm:7.14-dev .                                # once
scripts/build.sh                                          # -> build-rocm/ember-dflash
```

- `build/` is usually already configured, so `cmake --build build` is the fast
  inner loop.
- Every test is a plain binary with a `main()`; run it directly for full output
  or via ctest by name (ctest names drop the `test_` prefix):

  ```bash
  ./build/test_sse                       # direct: prints every FAIL line
  ctest --test-dir build -R sse -V       # by ctest name
  ctest --test-dir build --output-on-failure
  ```

- Tests use a hand-rolled `CHECK(cond, msg)` macro with `g_pass`/`g_fail`
  counters — no framework. **Adding a test file requires a new
  `add_executable` + `target_link_libraries(... ember_core m)` + `add_test`
  triple in the root `CMakeLists.txt`.**
- **Two CMake source lists must stay in sync.** `ember_core` (stub build) and
  the `ember-dflash` executable list every `src/**.c` explicitly. `ember-dflash`
  cannot link `ember_core` because that would collide `backend_stub.c` with
  `backend_dflash.cc`'s ABI symbols. A new `src/` file added to only one list
  builds fine on the host and fails (or silently misses code) in the ROCm build
  — this already caused one fix commit (`d8ace73`).
- `test_dspark_scheduler.cpp`, `test_thinking_budget.cpp`,
  `test_progress_cycle_detector.cpp`, and the `test_continuous_batch_*` /
  `test_resident_batch_coordinator.cpp` tests compile engine headers/sources
  directly (no `ember_core` link) so engine logic gets GPU-free coverage.

## Runtime verification (GPU-dependent — read before running)

`test/test_qa.c` covers the GPU-free runtime checklist. GPU-dependent release
verification requires exclusive access to a gfx1151 device and model weights;
the differential validator below is the supported end-to-end proof.

Before deploying an engine build, run the differential validator (in-process,
no production disruption beyond the GPU it needs):

```bash
./build-rocm/ember-dflash -m /models/model.gguf \
  --kv-cache-dir /tmp/ember-validation-cache \
  --validate-prompt prompt.txt --validate-tokens 32
```

It exits nonzero if greedy AR output diverges after snapshot restore, disk
round-trip, or (when DSpark is configured) on the speculative path. With
`--batch-sessions 2` it also verifies two resident sessions against the serial
baseline.

## Architecture: invariants you must not break

### The backend ABI is the seam

`src/backend/ember_backend.h` is the entire contract between the C server and
the model. The server never sees ggml or HIP. **Changing the ABI means changing
both implementations** (`backend_stub.c` and `backend_dflash.cc`), and the stub
must keep the whole pipeline exercisable without a GPU.

### Persistent generation workers, not HTTP connection threads

`http.c` is thread-per-connection, but connection threads hand chat jobs to
long-lived workers (`gen_worker` in `main.c`) and block. The default path has
one worker; `--batch-sessions N` creates N persistent workers. This is not
incidental: the DeepSeek4 compute-graph caches are `thread_local`, so a
short-lived thread rebuilt the ~918 MB prefill arena per request and orphaned
it on exit. Consequences that constrain any change here:

- `ember_kv_cache`, `ember_tool_memory`, and `ember_continuation_store` are
  shared and every access takes `state_lock`. Tool-memory readers return
  interior pointers that eviction can free, so callers must hold `state_lock`
  for the full lifetime of a borrowed pointer. The DSML tracker is request-local
  in `gen_ctx`, not shared.
- `ember_backend_free` and `ember_backend_release_idle_graphs` **must** be
  called on the worker thread (the caches live in its TLS). `main` must not
  free the backend — that would double-free.
- Idle reclaim (`EMBER_IDLE_RECLAIM_SECS`, default 300s) releases graphs from
  the worker's wait loop; the next request pays the rebuild.
- The job FIFO is bounded (`EMBER_MAX_QUEUE_DEPTH 8`) and sheds with 503.
  Foreground jobs jump ahead of `ember_background:true` jobs
  (`background_gate.c`).
- `/health`, `/status`, `/v1/models` never take `gen_lock`, so they stay
  responsive during a long generation. Keep it that way.

### Request pipeline order (`run_chat`, `src/server/main.c`)

Protocol adapter (`api_adapters.c`) → protocol-neutral `ember_chat_request` →
tool-memory replay attach → prompt render (`chat_template.c`) → splice-aware
encode → optional compaction → context guard → KV prefix lookup →
`ember_backend_generate` → SSE (`sse.c`) or buffered JSON response.

Two orderings are easy to break:

1. **Compaction runs *before* the context guard**, deliberately — a history
   that would 400 with `context_length_exceeded` gets served instead. After a
   successful compaction the prompt must be re-rendered and re-encoded through
   `encode_with_splices` (compaction returns messages, not tokens, precisely to
   preserve exact-DSML token identity).
2. **`usage.prompt_tokens` reports what the client sent**, captured before
   compaction and before an atomic malformed-tool-call retry grows the internal
   prompt. Server-authored recovery suffix tokens consume context/prefill but
   are never reported as completion tokens. Streaming requests do not hide a
   replacement assistant attempt inside the open response — matching ds4's
   `!stream` retry gate. When that leaves a malformed call unrecoverable the
   turn still finishes normally: ember drops the rejected block and stops with
   `finish_reason: "stop"` rather than emitting a typed error, because a
   malformed tool block is model output, not a server failure
   (`ds4_server.c:5231-5241`). Erroring cost a streaming agent the whole round.
   `EMBER_STREAM_TOOL_ERROR=1` restores the old error boundary.

### SSE: buffer-and-resplit, never incremental

`sse.c` keeps the full accumulated output and **re-splits it on every update**
(ds4's model). Lucebox's incremental holdback state machine broke five separate
ways (partial tool markers, split `</think>`, split emoji, prefill silence,
blind holdback). Any marker or codepoint split across *any* number of tokens
stays re-findable. Do not "optimize" this into an incremental emitter.
`test_sse.c` and `test_qa.c` fuzz every chunk size from 1 upward for exactly
this reason.

### KV cache: keys must come from real coverage

`kv_cache.c` holds no GPU state — it maps token prefixes to backend snapshot
slots. Subtleties:

- `ember_backend_snapshot_pos()` is authoritative for a snapshot's length,
  **not** the emitted token count: decode writes a token's KV row at the start
  of the *next* step, so a post-generation snapshot lags the stream by one row
  (and speculative decode can run ahead). A key longer than its KV describes a
  prefix the snapshot cannot honor.
- Slot `EMBER_KV_MAX_SLOTS - 1` (63) is reserved as the disk-restore staging
  slot (`EMBER_KV_DISK_SLOT`); the in-memory cache occupies slots
  `[0, --prefix-cache-slots)` (default 8) and never overlaps it.
- The logical prefix entry is committed only when the backend reports
  `snapshot_saved`, so a failed save cannot poison the cache.

### Exact-token tool replay

DSML markers are special tokens that do **not** survive detokenize→retokenize.
`tool_memory.c` stores the exact sampled bytes *and* token ids per tool-call
id; the renderer emits a splice sentinel and `encode_with_splices` splices the
stored ids verbatim. That token identity is what lets a post-tool-call KV
snapshot continue instead of re-prefilling. `continuation.c` persists the same
frontier across restarts for ID-only continuations. The rationale is documented
in `src/model/tool_memory.h` and
`src/model/continuation.h`. Call-id/frontier arrays are dynamic; do not
reintroduce a silent fixed parallel-call ceiling.

### Progress signals are request-local; recovery is opt-in

Ember deliberately has no fixed tool-round cap, matching ds4's
`ds4_agent.c:8448`: round count cannot distinguish a legitimate long agent run
from a loop. Ember derives additive diagnostics from full request history:
`ember_chat_request_tool_loop_rounds()` compares call/result rounds,
`ember_chat_request_tool_loop_calls()` compares call signatures, and
`ember_chat_request_progress_lease()` counts trailing rounds with no novel
`(tool name, exact result)` effect. Reporting may add metadata, logs, and
`/status` telemetry, but must not change `finish_reason:"tool_calls"`, refuse a
request, or become cross-request detection state. `--auto-answer-after-loop` is
the explicit, behavior-changing exception: it is off by default, suppresses
optional tools for one turn, adds a private recovery instruction, and never
overrides required tool choice. Continuation-only histories and multi-round
cycles remain documented limitations.

### `engine/` is a vendored fork — treat it as such

`engine/` is a snapshot of lucebox (`ggml` fork with ROCMFP kernels + the
DeepSeek4 backend), pinned in `engine/VENDOR.md` with its origin commit. Port
upstream fixes by diffing against that commit, and update `VENDOR.md` when the
snapshot moves. Local engine changes are legitimate (recent commits touch
DSpark scheduling and batched verification) but they are fork divergence — say
so in the commit message. One measured engine decision to respect: HIP graph
replay stays **OFF** (A/B-measured regression on gfx1151; see the long comment
in `engine/CMakeLists.txt` — do not re-enable until the graph key is stable).

## Conventions

- C11 for the server, C++17 for the bridge and engine. `-Wall -Wextra`.
- `ember_buf` (`src/common/buf.h`) is the universal growable buffer, always
  NUL-terminated. **Allocation failure aborts** (`ember_buf_fatal`) — matching
  Dwarfstar's fail-fast policy, because continuing would emit a truncated but
  syntactically valid protocol payload. Do not add `NULL` checks that try to
  recover.
- Every exported symbol is `ember_`-prefixed. Headers carry the *why* — a design
  rationale block at the top of each `.h` is the norm; read it before editing
  the `.c`.
- Chat/default HTTP errors are OpenAI-shaped
  (`{"error":{"message","type","code"}}`); Responses streaming and Anthropic
  errors retain their native protocol envelopes. All are JSON-escaped through
  `ember_json_escape`.
- New behaviour ported from ds4/Dwarfstar or lucebox: cite the source location
  in a comment (e.g. `ds4_agent.c:8010-8292`) — the existing code does this
  consistently and it is how parity gets audited.
- Risky parity features ship **off by default** (`--auto-compact`, DSpark's
  confident-prefix rule) and are enabled in production explicitly.
- Commit messages: `type(scope): subject` — e.g. `feat(server)`, `fix(engine)`,
  `perf(engine)`, `refactor(server)`, `fix(build)`, `docs`.

## Testing strategy

- The ctest gauntlet (GPU-free) covers: SSE fuzzing, tool parser/schema/memory,
  continuation, DSML decode, JSON, chat API, API adapters, HTTP, background
  gate, chat template, compaction, GGUF, model card, KV cache, QA behaviours
  (`test_qa.c`), plus C++ engine tests (prefill policy, DSpark scheduler,
  thinking budget, progress cycle detector, continuous batch
  scheduler/executor, resident batch coordinator).
- Python tests run through ctest too: server-level integration
  (`test_continuous_batch_server.py`, `test_tool_safety_server.py`, which spawn
  the real `ember-server` binary) and the quant pipeline
  (`test_quant_quality_report.py`, `test_quant_behavior_eval.py`,
  `test_gguf_tensor_error.py`, `test_quant_manifest_corpus.py`). All Python is
  stdlib-only.
- GPU-dependent runtime validation requires exclusive access to a target GPU.

## Container deployment

- `docker compose up --build` builds the server, downloads the default published
  model when needed, and persists model and KV data in local mounted directories.
- The `dev` target is AMD's stock
  `rocm/dev-ubuntu-24.04:7.14.0-full` plus build tooling, source, and symbols.
  The Ubuntu-based `release` target contains only the stripped server, its
  recursive ROCm ELF dependency closure, rocBLAS runtime kernel data, download
  utilities, and `libsegvtrace.so`. The shim is LD_PRELOAD'd to print a
  symbolized backtrace on fatal signals because a real core dump is impractical
  at ~100 GB RSS.
- `scripts/build.sh` uses the `dev` image and does not require a GPU merely to
  compile because `gfx1151` is pinned explicitly. Override the image with
  `EMBER_IMAGE` and parallelism with `JOBS`.
- Key runtime env vars (`.env.example` lists container settings):
  `DFLASH_DS4_SPEC=1` +
  `DFLASH_DS4_DRAFT=<draft.gguf>` enable DSpark; `EMBER_BG_IDLE_SECS` /
  `EMBER_BG_MAX_WAIT_SECS` tune background gating; `EMBER_IDLE_RECLAIM_SECS`
  controls idle graph reclamation; `EMBER_TRACE_TOKENS` and
  `DFLASH_TRACE_SAMPLER` enable off-by-default token/sampler forensics;
  `DS4_SERVER_PREFILL_QUANTUM`,
  `DS4_SERVER_MIXED_PREFILL_QUANTUM`, `DS4_SERVER_DECODE_COALESCE_US` tune
  batched-mode scheduling.

## Security considerations

- The server binds to loopback (`--port`, default 8080); browser CORS is off
  unless `--cors` is passed.
- Hardening already in place, preserve it: socket send/recv timeouts so a stuck
  client cannot wedge a slot; 64 MB request-body cap; query-string stripping;
  JSON-escaped output everywhere; validated surrogate pairs.
- Tool calls pass an executable-validation boundary: incomplete repair, nested
  or mixed protocol markup, malformed raw JSON, duplicate keys, unknown or
  tool-choice-excluded names, forbidden parallel calls, non-object arguments,
  and recursive JSON-Schema violations are rejected; strict schemas fail closed
  on unsupported keywords. Repeated
  violations return typed errors (`422 invalid_tool_call` / one terminal
  `model_output_error` SSE event) without exposing partial calls.
- Persisted tool-call replay is model-scoped and bound to call
  ID/`previous_response_id` frontier bindings, so a tool id cannot replay a
  hidden or force-closed assistant trajectory.
- The compaction prompt is private: compaction turns never snapshot KV, so the
  summary prompt cannot become a cache entry a later turn restores from.
- Do not read or transmit secrets; the repo carries none. Model paths
  (`/models`) and the KV cache dir are local operator territory.

## When in doubt

- Parity question (streaming, tool calling, reasoning, KV semantics)? Read the
  rationale headers and regression tests first; several divergences from ds4
  are deliberate and must not be reverted without measurement.
- Performance question on gfx1151? Measure. The project culture is "MEASURED,
  not assumed" (see the HIP-graph-replay comment in `engine/CMakeLists.txt` and
  the `perf(engine): revert` commit history).
