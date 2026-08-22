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

The opt-in `release-xdna` prototype adds a second provider seam below the
backend ABI. Its measured Gen53 placement keeps the target and authoritative
q=1 prefix verifier on the GPU, runs resident DSpark projection/shared-expert
work on XDNA2, and runs routing plus ROCMFP4 experts through AVX-512 CPU code.
Its 100-prompt and 15-case quality corpora pass, but it is not release-default:
representative serial throughput remains 25.7% below target-only.

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
providers/xdna2/ Optional XRT provider, CPU SIMD quant kernels, AIE/IRON
               sources, validators, and pinned provenance. This remains below
               the backend ABI and must keep ordinary HIP fallback intact.
test/          Plain C/C++ test binaries (one main() each) plus Python
               server-level and quant-pipeline tests.
scripts/       build.sh (ROCm container build), diagnostics, and the quant
               quality pipeline (*.py, stdlib-only).
share/         model_cards/ (per-model defaults sidecar JSON + _schema.json),
               quant_eval/ (eval fixtures).
reports/       Generated quant quality reports (Markdown/JSON/CSV/SVG).
docker/        Multi-stage Dockerfile: full-ROCm `dev` toolchain and minimal
               dependency-closure `release` image (published through GHCR).
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
  `add_executable` + `target_link_libraries` + `add_test` triple in the root
  `CMakeLists.txt`**, and the new ctest name must be added to `EMBER_C_TESTS`
  (which sets the 60s `TIMEOUT`). Link `ember_core`, plus `m` and/or `xgrammar`
  as the test needs them — a bare `ember_core` is the common case. An
  ember-owned target must also join `EMBER_STRICT_TARGETS`; see CI gates below.
- **Two CMake source lists must stay in sync.** `ember_core` (stub build) and
  the `ember-dflash` executable list every `src/**.c` explicitly. `ember-dflash`
  cannot link `ember_core` because that would collide `backend_stub.c` with
  `backend_dflash.cc`'s ABI symbols. A new `src/` file added to only one list
  builds fine on the host and fails (or silently misses code) in the ROCm build
  — this already caused one fix commit (`d8ace73`). `ci/check_invariants.py`
  enforces this mechanically; a file legitimately in one list only needs an
  entry in that script's `SINGLE_LIST_EXCEPTIONS` with a reason.
- A growing set of C++ tests compiles engine, `providers/xdna2/`, or other
  vendored sources directly instead of linking `ember_core`, so that logic gets
  GPU-free coverage: `test_prefill_policy`, `test_dspark_scheduler`,
  `test_thinking_budget`, `test_progress_cycle_detector`, `test_sampler`,
  `test_pre_tokenizer`, `test_continuous_batch_{scheduler,executor}`,
  `test_resident_batch_coordinator`, and the `test_xdna_*` set. These are
  exactly the targets held *out* of `EMBER_STRICT_TARGETS` — upstream keeps its
  own warning standard, and a directory-scoped strict flag would turn every
  fork sync into a warning-fixing exercise.

## Runtime verification (GPU-dependent — read before running)

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
round-trip, or (when DSpark is configured) on the speculative path. With
`--batch-sessions 2` it also verifies two resident sessions against the serial
baseline.

For releases, GitHub CI performs this validation automatically on the dedicated
gfx1151 runner after candidate publication. It checks IOMMU and the pinned model
pair, quiesces the configured production container for exclusive GPU access,
and restores production even when certification fails.

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
- Python tests run through ctest too, and are registered only when CMake finds
  a Python 3 interpreter. Four spawn the real `ember-server` binary and carry a
  tighter 20s timeout: `test_continuous_batch_server.py`,
  `test_tool_safety_server.py`, `test_request_budgets_server.py`,
  `test_client_compatibility_server.py`. The rest are offline analysis — the
  quant pipeline (`test_quant_quality_report.py`, `test_quant_behavior_eval.py`,
  `test_gguf_tensor_error.py`, `test_quant_manifest_corpus.py`,
  `test_resident_benchmark.py`) and the release tooling
  (`test_release_scripts.py`, `test_release_changelog.py`,
  `test_mirror_gh_issues.py`). All Python is stdlib-only; there is no
  `pyproject.toml` and no dependency install step.
- GPU-dependent runtime validation requires exclusive access to a target GPU.
  XDNA claims additionally require the pinned host driver/firmware/XRT tuple,
  translated IOMMU domains, `/dev/accel/accel0`, provider validators, and the
  trained two-session differential/throughput gates.

## CI gates

Every gate below is GPU-free and reproducible locally — none of them need
ROCm, a gfx1151 device, or model weights. They run on push and PR through
`.forgejo/workflows/ci.yml`, ordered cheapest-first, and are mirrored in
`.github/workflows/ci.yml`. A change can build and pass `ctest` locally and
still fail three of these, so run them before pushing anything non-trivial.

1. **Repo invariants** — `python3 ci/check_invariants.py`. Catches four things
   a compiler cannot: a `src/` file added to only one of the two hand-maintained
   CMake source lists (the ROCm-build-only failure mode, `d8ace73`); a test that
   compiles but was never registered with ctest, which reads as coverage that
   does not exist; an `add_test()` with no `TIMEOUT`, which lets a hang block
   ctest forever rather than fail; and a target that links `ember_core` but is
   missing from `EMBER_STRICT_TARGETS`. Takes under a second — run it after any
   `CMakeLists.txt` edit.

2. **Strict build and test**, in both `Release` *and* `Debug`:

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DEMBER_STRICT=ON
   cmake --build build -j"$(nproc)" && ctest --test-dir build --output-on-failure
   ```

   `EMBER_STRICT=ON` adds `-Werror -Wshadow -Wconversion -Wsign-conversion
   -Wformat=2 -Wnull-dereference`. `src/` compiles with zero warnings under it,
   so it is a real gate rather than an aspiration. It is deliberately opt-in
   locally (an in-progress edit should not be blocked by a warning) and applied
   per-target via `EMBER_STRICT_TARGETS`, never via `add_compile_options` —
   see the build-and-test section on which targets are excluded and why. Debug
   matters independently of Release: the assertions and the different inlining
   reach paths `-O3` folds away.

3. **Analyzers** — both are release gates, and every new warning fails the job
   rather than growing a baseline. Reviewed GCC false positives carry narrow
   in-source suppressions with an ownership rationale:

   ```bash
   for f in src/common/*.c src/model/*.c src/server/*.c src/backend/backend_stub.c; do
     gcc -fanalyzer -c -o /dev/null -std=c11 -Wall -Wextra -Werror -Isrc -D_GNU_SOURCE "$f"
   done
   cppcheck --enable=warning,portability --inline-suppr --std=c11 \
            --error-exitcode=1 --suppress=missingIncludeSystem \
            -I src src/common src/model src/server src/backend/backend_stub.c
   ```

4. **Per-file coverage ratchet** — `ci/coverage_floors.json` pins a line-coverage
   floor for each file individually and only ever moves up. An aggregate
   percentage would let a new untested 500-line module land unnoticed; the
   flip side is that adding uncovered lines to an already well-covered file
   (`sse.c` is at 94.6) fails CI even though the project total barely moves.
   Reproduce with a `--coverage -O0 -g` build tree and
   `python3 ci/coverage.py --build build-cov`.

5. **Sanitizers (ASan + UBSan + LSan)** — an authoritative gate, but it runs
   **only** in `.github/workflows/ci.yml`, not Forgejo. This is deliberate and
   documented at the top of `.forgejo/workflows/ci.yml`: Forgejo 10's Docker
   exec path on that runner hangs LeakSanitizer teardown and turns ctest's
   timeout signal into an unbounded `AddressSanitizer:DEADLYSIGNAL` log, while
   the same commits pass under GitHub. Do not "fix" the gap by duplicating the
   job into Forgejo. Locally:

   ```bash
   cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
     -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
     -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
   ```

The container workflow (`container.yml`) builds the ROCm `dev` and `release`
images without a GPU; GPU runtime validation is separate and covered under
runtime verification above. `docs/ci.md` is the long-form reference.

## Container deployment

- `docker compose up -d` pulls the immutable GHCR release image, downloads the
  default model when needed, and persists model and KV data in local mounted
  directories. `compose.build.yaml` is the explicit local source-build override.
- The `dev` target is AMD's stock
  `rocm/dev-ubuntu-24.04:7.14.0-full` plus build tooling, source, and symbols.
  The Ubuntu-based `release` target contains only the stripped server, its
  recursive ROCm ELF dependency closure, rocBLAS runtime kernel data, download
  utilities, and `libsegvtrace.so`. The shim is LD_PRELOAD'd to print a
  symbolized backtrace on fatal signals because a real core dump is impractical
  at ~100 GB RSS.
- The experimental `release-xdna` target layers an XRT userspace stack, XDNA
  shim, AOT IRON artifacts, and the opt-in target-MoE/DSpark provider over
  `release`. `compose.xdna.yaml` enables only the measured resident DSpark
  placement with two sessions; the older target-expert placement is research,
  not a serving optimization. The `amdxdna` kernel driver, firmware, and IOMMU
  setup remain host responsibilities. This target is not release-default.
- `scripts/build.sh` uses the `dev` image and does not require a GPU merely to
  compile because `gfx1151` is pinned explicitly. Override the image with
  `EMBER_IMAGE` and parallelism with `JOBS`.
- Release CI reuses a stable BuildKit instance, a BuildKit-mounted 20 GiB
  ccache, hosted/Forgejo compiler caches, and a persistent Trivy database. Do
  not replace the stable builder with a per-run name. Hardware certification
  uses disposable KV state and caches successful model-integrity checks for
  seven days, invalidated by file identity/metadata changes. Cache-miss hashing
  must use direct I/O: buffered reads were measured to starve the following UMA
  allocation even after `POSIX_FADV_DONTNEED` on the model XFS volume.
  Certification must stop and restore the supervising `ember-server.service`
  through the fixed-purpose host wrapper; stopping its child container alone
  causes systemd to recreate the 90 GiB process during validation.
- Key runtime env vars (`.env.example` lists container settings):
  `DFLASH_DS4_SPEC=1` +
  `DFLASH_DS4_DRAFT=<draft.gguf>` enable DSpark; `EMBER_BG_IDLE_SECS` /
  `EMBER_BG_MAX_WAIT_SECS` tune background gating; `EMBER_IDLE_RECLAIM_SECS`
  controls idle graph reclamation; `EMBER_TRACE_TOKENS` and
  `DFLASH_TRACE_SAMPLER` enable off-by-default token/sampler forensics;
  `DFLASH_DSPARK_XDNA_PLUGIN`, `DFLASH_DSPARK_XDNA_GPU_MAIN`, and
  `DFLASH_DSPARK_XDNA_REQUIRED` control the optional resident NPU proposal path;
  `DS4_SERVER_PREFILL_QUANTUM`,
  `DS4_SERVER_MIXED_PREFILL_QUANTUM`, `DS4_SERVER_DECODE_COALESCE_US` tune
  batched-mode scheduling.

## Security considerations

- The server binds to loopback by default (`--host 127.0.0.1`, `--port 8080`);
  an explicit `--host 0.0.0.0` exposes its unauthenticated API. Browser CORS is
  off unless `--cors` is passed.
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
