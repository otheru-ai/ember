# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Ember is a from-scratch **C** inference server for DeepSeek-V4-Flash on AMD Strix
Halo (gfx1151). It is
**ds4/Dwarfstar's server architecture rewritten clean in C, driving lucebox's
tuned HIP kernels** through a stable C ABI. See `README.md` for installation
and first use, and `ARCHITECTURE.md` for the layering rationale.

The load-bearing decision: the GPU kernels and the tokenizer are *reused*
(person-years of gfx1151 tuning, and a `joyai-llm` pre-tokenizer that must be
byte-exact); everything above the forward pass is *rewritten fresh in C*.

The opt-in heterogeneous prototype remains below the same ABI: the GPU owns the
target and authoritative q=1 prefix verifier, XDNA2 owns resident DSpark
projection/shared-expert runlists, and AVX-512 CPU code owns draft routing and
ROCMFP4 routed experts. Its 100-prompt and 15-case quality corpora pass, but it
is not release-default because representative serial throughput remains 25.7%
below target-only.

## Build & test

Two build configurations. Almost all work happens in the first one.

```bash
# GPU-free: server + stub backend + full test gauntlet. Builds on any host.
cmake -S . -B build && cmake --build build && ctest --test-dir build

# Real backend (ROCm/HIP; must run in the container — no HIP toolchain on host)
docker build --target dev -f docker/Dockerfile \
  -t ember-rocm:7.14-dev .                                # once
scripts/build.sh                                          # -> build-rocm/ember-dflash
```

The opt-in `release-xdna` Docker target packages the pinned XRT runtime, IRON
artifacts, and resident DSpark provider. It requires the host `amdxdna` driver,
firmware, enabled IOMMU, and `/dev/accel/accel0`; use `compose.xdna.yaml`. The
overlay selects the measured two-session DSpark placement, while ordinary HIP
remains the normal release path and fallback.

`build/` is already configured, so `cmake --build build` is the fast inner loop.

Single test — every test is a plain binary with a `main()`; run it directly for
full output, or via ctest by name:

```bash
./build/test_sse                       # direct: prints every FAIL line
ctest --test-dir build -R sse -V       # by ctest name (no `test_` prefix)
ctest --test-dir build --output-on-failure
```

Tests use a hand-rolled `CHECK(cond, msg)` macro with `g_pass`/`g_fail`
counters — no framework. Adding a test file requires a new
`add_executable` + `target_link_libraries(... ember_core m)` + `add_test` triple
in `CMakeLists.txt`.

**Two CMake source lists must stay in sync.** `ember_core` (stub build) and the
`ember-dflash` executable list every `src/**.c` explicitly. `ember-dflash`
cannot use `ember_core` because that links `backend_stub.c`, which would collide
with `backend_dflash.cc`'s ABI symbols. A new `src/` file added to only one list
builds fine on the host and fails (or silently misses code) in the ROCm build —
this has already caused one fix commit (`d8ace73`).

`test_dspark_scheduler.cpp` is the exception: it links nothing, compiling a
header-only engine class (`engine/dflash/deepseek4/deepseek4_dspark_scheduler.h`)
directly so engine scheduling logic gets GPU-free coverage.

## Runtime verification

`test/test_qa.c` covers the GPU-free runtime checklist. GPU-dependent release
verification requires exclusive access to a gfx1151 device and model weights;
the differential validator below is the supported end-to-end proof.

Before deploying an engine build manually, run the differential validator:

```bash
./build-rocm/ember-dflash -m /models/model.gguf \
  --kv-cache-dir /tmp/ember-validation-cache \
  --validate-prompt prompt.txt --validate-tokens 32
```

It exits nonzero if greedy AR output diverges after snapshot restore, disk
round-trip, or (when DSpark is configured) on the speculative path.

For releases, GitHub CI performs this validation automatically on the dedicated
gfx1151 runner after candidate publication. It checks IOMMU and the pinned model
pair, quiesces the configured production container for exclusive GPU access,
and restores production even when certification fails.

## Architecture: what you must know before editing

### The backend ABI is the seam

`src/backend/ember_backend.h` is the entire contract between the C server and
the model. Two implementations: `backend_stub.c` (deterministic, GPU-free, drives
every test) and `backend_dflash.cc` (`extern "C"` shim over the vendored engine).
The server never sees ggml or HIP. **Changing the ABI means changing both
implementations**, and the stub must keep the pipeline exercisable without a GPU.

### Persistent generation workers, not HTTP connection threads

`http.c` is thread-per-connection, but connection threads hand chat jobs to
long-lived workers (`gen_worker` in `main.c`) and block. The default path has
one worker; `--batch-sessions N` creates N persistent workers. This is not
incidental: the DeepSeek4 compute-graph caches are `thread_local`, so a
short-lived thread rebuilt the ~918 MB prefill arena per request and *orphaned*
it on exit. Consequences that constrain any change here:

- `ember_kv_cache`, `ember_tool_memory`, and `ember_continuation_store` are
  shared and every access takes `state_lock`. Tool-memory readers return
  interior pointers that eviction can free, so callers must hold `state_lock`
  for the full lifetime of a borrowed pointer. The DSML tracker is request-local
  in `gen_ctx`, not shared.
- `ember_backend_free` and `ember_backend_release_idle_graphs` **must** be called
  on the worker thread (the caches live in its TLS). `main` must not free the
  backend — that would double-free.
- Idle reclaim (`EMBER_IDLE_RECLAIM_SECS`, default 300s) releases graphs from the
  worker's wait loop; the next request pays the rebuild.
- `EMBER_TRACE_TOKENS` and `DFLASH_TRACE_SAMPLER` enable off-by-default
  token-byte and post-filter sampler forensics; keep both disabled normally.
- The job FIFO is bounded (`EMBER_MAX_QUEUE_DEPTH 8`) and sheds with 503.
  Foreground jobs jump ahead of `ember_background:true` jobs (`background_gate.c`).
- `/health`, `/status`, `/v1/models` never take `gen_lock`, so they stay
  responsive during a long generation. Keep it that way.

### Request pipeline (`run_chat`, `src/server/main.c` — the 2.1k-line hub)

Protocol adapter (`api_adapters.c`: Responses / Anthropic / legacy Completions)
→ protocol-neutral `ember_chat_request` → tool-memory replay attach → prompt
render (`chat_template.c`) → splice-aware encode → optional compaction →
context guard → KV prefix lookup → `ember_backend_generate` → SSE
(`sse.c`) or buffered JSON response.

Order matters in two places that are easy to break:

1. **Compaction runs *before* the context guard**, deliberately — a history that
   would 400 with `context_length_exceeded` gets served instead. After a
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

`sse.c` keeps the full accumulated output and **re-splits it on every update**.
This is ds4's model, adopted deliberately: lucebox's incremental holdback state
machine broke five separate ways (partial tool markers, split `</think>`, split
emoji, prefill silence, blind holdback). Any marker or codepoint split across
*any* number of tokens stays re-findable. Do not "optimize" this into an
incremental emitter. `test_sse.c` and `test_qa.c` fuzz every chunk size from 1
upward for exactly this reason.

### KV cache: keys must come from real coverage

`kv_cache.c` holds no GPU state — it maps token prefixes to backend snapshot
slots and decides where to cut. Two subtleties:

- `ember_backend_snapshot_pos()` is authoritative for a snapshot's length, **not**
  the emitted token count: decode writes a token's KV row at the start of the
  *next* step, so a post-generation snapshot lags the stream by one row (and
  speculative decode can run ahead). A key longer than its KV describes a prefix
  the snapshot cannot honor.
- Slot `EMBER_KV_MAX_SLOTS - 1` (63) is reserved as the disk-restore staging slot
  (`EMBER_KV_DISK_SLOT`); the in-memory cache occupies slots
  `[0, --prefix-cache-slots)` (default 8) and never overlaps it.
- The logical prefix entry is committed only when the backend reports
  `snapshot_saved`, so a failed save cannot poison the cache.

### Exact-token tool replay

DSML markers are special tokens that do **not** survive
detokenize→retokenize. `tool_memory.c` therefore stores the exact sampled bytes
*and* token ids per tool-call id; the renderer emits a splice sentinel and
`encode_with_splices` splices the stored ids verbatim. That token identity is
what lets a post-tool-call KV snapshot continue instead of re-prefilling. See
`src/model/tool_memory.h`. `continuation.c` persists the same frontier
across restarts for ID-only continuations. Call-id/frontier arrays are dynamic;
never reintroduce a silent fixed parallel-call ceiling. Executable calls are
also checked for unambiguous markup, duplicate keys, tool-choice/parallel-call
constraints, and recursive JSON-Schema conformance before any frame is emitted.

Ember also deliberately has no fixed tool-round ceiling, matching ds4's
`ds4_agent.c:8448`. It derives additive full-history diagnostics from repeated
call signatures, identical call/result rounds, and trailing tool rounds with no
novel `(tool name, exact result)` effect. Reporting ignores call ids, never
changes `tool_calls`/`tool_use`, never refuses generation, and never uses
`/status` telemetry as cross-request detection state. The explicit exception is
the opt-in `--auto-answer-after-loop`: it suppresses optional tools for one turn
and adds a private plain-prose recovery instruction, but never overrides
required tool choice. Continuation-only histories and multi-round cycles are
documented limitations.

### `engine/` is a vendored fork — treat it as such

`engine/` is a snapshot of lucebox (`ggml` fork with ROCMFP kernels + the
DeepSeek4 backend), pinned in `engine/VENDOR.md` with its origin commit. Port
upstream fixes by diffing against that commit, and update `VENDOR.md` when the
snapshot moves. Local engine changes are legitimate (recent commits touch DSpark
scheduling and the CPU fp2 path) but they are fork divergence — say so in the
commit message.

## Conventions

- C11 for the server, C++17 for the bridge and engine. `-Wall -Wextra`.
- `ember_buf` (`src/common/buf.h`) is the universal growable buffer, always
  NUL-terminated. **Allocation failure aborts** (`ember_buf_fatal`) — matching
  Dwarfstar's fail-fast policy, because continuing would emit a truncated but
  syntactically valid protocol payload. Do not add `NULL` checks that try to
  recover.
- Every exported symbol is `ember_`-prefixed. Headers carry the *why* — a design
  rationale block at the top of each `.h` is the norm, and worth reading before
  editing the `.c`.
- Chat/default HTTP errors are OpenAI-shaped
  (`{"error":{"message","type","code"}}`); Responses streaming and Anthropic
  errors retain their native protocol envelopes. All are JSON-escaped through
  `ember_json_escape`.
- New behaviour ported from ds4/Dwarfstar or lucebox: cite the source location in
  a comment (e.g. `ds4_agent.c:8010-8292`) — the existing code does this
  consistently and it is how parity gets audited.
- Risky parity features ship **off by default** (`--auto-compact`,
  DSpark's confident-prefix rule) and are enabled in production explicitly.
- Commit messages: `type(scope): subject` — `feat(server)`, `fix(engine)`,
  `perf(engine)`, `build`, `debug(engine)`.
