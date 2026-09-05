# Independent review: nested DSML boundaries at b26ed71

Reviewed `b26ed7126582842ebba3fd06e0639205d4047428` on 2026-09-05 in an isolated
worktree. **Approved for the host-side parser/boundary repair.** No remaining
blocking finding in this revision. This closes the three findings against
`a9f1d635f04052569eeaaf624fadcb6f14e22020`; that original review remains preserved
at `080d53a`. This approval does not activate the dark flag or establish model
quality, production behavior, hardware performance, or release readiness.

## Independent contract evidence

The standalone [contract probe](evidence/b26ed71/contract.c) constructs intended
argument values independently of the close matcher. It compares both parsed
arguments and captured SSE argument bytes against that literal-value JSON oracle.
No named tool is executed. Across all four DSML spellings it exercises ordinary
Unicode text, nested parameters, nested invokes, complete nested wrappers, and
a different DSML family nested inside a value. Each call also has a following
parameter with a sentinel value, so consuming or dropping subsequent arguments
fails the check.

For accepted cases, every byte prefix must avoid a premature tool boundary, and
the complete boundary must equal the end of the outer wrapper. For every fixed
chunk size from one byte through the full message length, SSE must emit nothing
before validation, then emit one outer call with exactly the intended arguments.
With the flag absent, nested cases must be refused and remain un-emitted.
This is every fixed-size chunking of these inputs, not every arbitrary partition
or every possible malformed input.

| Build/revision | Flag absent | Flag enabled |
|---|---:|---:|
| Preserved a9f1d63 negative control | 16288 checks, 0 failures | 23048 checks, 4852 failures |
| b26ed71 strict Debug | 16288 checks, 0 failures | 23048 checks, 0 failures |
| b26ed71 ASan/UBSan/LSan | 16288 checks, 0 failures | 23048 checks, 0 failures |

The same source probe was linked separately against each revision's own core
library. The old enabled-mode failure is expected and demonstrates that the
probe detects the defects rather than merely mirroring the new implementation.
The project's registered parser tests additionally exercise unbalanced values,
structural nesting, mixed structural families, malformed attributes, invalid JSON,
and truncated/repaired blocks in both flag modes.

## Findings closed

1. `src/server/sse.c:123` now obtains the outer tool boundary through the same
   matching implementation used by validation. The independent prefix check
   reaches the inner-wrapper counterexample and preserves the outer boundary.
   The four production call sites in `src/server/main.c` are generation stop
   (444), snapshot eligibility (834), initial remembered span (958), and the
   remembered sibling-span extension (984). They all call this helper. Boundary
   checks cover the helper; no live model/tokenizer snapshot replay is claimed.
2. The invoke and parameter walks at `src/server/sse.c:422,430` share the matching
   implementation. The parameter-only counterexample now emits the complete
   validated argument, including the inner closer and trailing sentinel. The
   broader contract also checks nested invokes and full wrappers.
3. Both `tool_parser_nested` and `sse_nested` are registered and inherit the
   timeout list. `python3 ci/check_invariants.py` passes on the unmodified target.

## Non-blocking qualification to wire 1072

Sharing the closing matcher fixes these demonstrated disagreements, but it does
not establish that the parser and emitter can never diverge: they retain separate
outer walks and control flow. In this revision `EMBER_SSE_TOOL` buffers tool bytes;
`emit_tool_stream` is called by the validation-gated `ember_sse_emit_tools` flush.
The claim that a second incremental walker is structurally necessary because
validation occurs at the end is therefore unsupported. Emitting validated call
objects is a possible future design. A redesign is not required for this approval;
the shared matcher plus independent byte-equality checks closes the reported bug.

## Host verification

All builds below were configured and built by Codex in this isolated worktree;
the binaries are owned by the local `mythos` account and were built during this
review. The owner worktree and production were not changed.

- Strict Debug: 73/73 tests pass.
- Strict Release: 73/73 tests pass.
- ASan, UBSan and LSan: 73/73 tests pass, plus both contract modes.
- GCC `-fanalyzer` across the documented host source set: no diagnostics.
- cppcheck warning/portability gate: pass.
- Coverage suite: 73/73 tests pass; all per-file floors pass. Parser 88.9%, SSE 95.7%.
- Repository invariants: pass.

Sanitizer executables use `-fno-pie` and `-no-pie` to avoid the previously
independently reproduced host ASan/PIE startup instability; address, undefined
behavior, and leak checks remain enabled. No GPU or model execution was used.

Raw suite logs, probe outputs, and SHA-256 hashes are in
[the evidence manifest](evidence/b26ed71/manifest.json).

Reproduce the independent probe after a strict Debug host build:

```sh
cc -std=c11 -Wall -Wextra -Werror -Wshadow -Wconversion -Wsign-conversion \
  -Isrc docs/reviews/evidence/b26ed71/contract.c \
  build-review-debug/libember_core.a -o /tmp/nested-parser-contract
env -u EMBER_DSML_NESTED_VALUES /tmp/nested-parser-contract
EMBER_DSML_NESTED_VALUES=1 /tmp/nested-parser-contract
```

For sanitizers, build the core with the documented sanitizer flags, add
`-fno-pie` to C/C++ compilation and `-no-pie` to executable linking, and compile
this probe with `-fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie
-no-pie`. Run both modes with `ASAN_OPTIONS=detect_leaks=1`.
