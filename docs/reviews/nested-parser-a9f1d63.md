# Independent review of a9f1d63

**Changes required before enabling the nested-value feature.** Review target:
`a9f1d635f04052569eeaaf624fadcb6f14e22020`, currently part of the pending release
bundle. The parser-only tests pass, but the generation stop and SSE emission
paths still use first-close matching. One path truncates the generated block;
the other can deliver different tool arguments from those that passed validation.
The feature remains off by default. No production flags, runner jobs or GPU
operations were changed during this review.

## P1 — stopping generation at the inner wrapper prevents recovery

`src/server/sse.c:119` implements `ember_find_tool_end()` using the first closing
tag of the selected family. `gen_token`, `src/server/main.c:444`, uses that
pointer to truncate the accumulated output and halt generation. This runs for
both streaming and collected output.

The independent 318-byte fixture contains a balanced tool-call block inside a
string argument. With `EMBER_DSML_NESTED_VALUES=1`, parsing the complete text
returns one valid outer call. `ember_find_tool_end()` returns byte **260**, the
end of the inner wrapper. The resulting prefix parses as zero calls,
`complete=false`, `malformed=true`. Thus the full text used by the new parser
test is not reachable through this normal stop boundary.

Make the generation boundary agree with the nested-value grammar. The same
helper is used for exact-token replay capture at `main.c:958`, so repairing only
the callback would leave the saved tool span truncated. Preserve existing
stop precedence and the special sibling-block handling of native formats.

## P1 — SSE changes arguments after validation

The complete call is validated through `parse_executable_tool_calls()` before
`ember_sse_emit_tools()` is invoked (`main.c:2975` and the corresponding native
adapter path). But `emit_tool_stream()` re-parses the original bytes with first
invoke/parameter closers at `sse.c:413` and `sse.c:420`. It does not use the
arguments that passed validation. The new nested parser therefore admits a
value the emitter subsequently shortens.

This defect is independently reachable even without an inner tool-call wrapper.
The second fixture has just a nested parameter inside `content`; generation's
wrapper end is correctly byte **229 of 229**, and the full parser accepts it.
Its validated argument is:

```json
{"content":"BEFORE <?DSML?parameter name=\"inner\" string=\"true\">nested</?DSML?parameter> AFTER"}
```

The SSE sink receives:

```json
{"content":"BEFORE <?DSML?parameter name=\"inner\" string=\"true\">nested"}
```

One outer call is emitted, but the content is truncated into a different,
still-valid JSON argument. The first 318-byte fixture also reproduces this if
the complete text reaches the emitter. This violates the executable-validation
boundary, rather than merely failing to salvage a malformed model output.

Emit the validated call objects, or share the same authoritative parsing logic
and prove argument identity across that boundary. A post-validation emitter
must not independently reinterpret the accepted payload. Add coverage that
reassembles the actual emitted arguments and compares the complete JSON values,
including their closing tags and trailing text; checking for an interior
substring such as `echo nested` would miss this failure.

## P2 — the new test fails the repository invariant gate

`tool_parser_nested` is registered at `CMakeLists.txt:134` without a timeout and
is absent from `EMBER_C_TESTS` at line 453. `python3 ci/check_invariants.py`
exits 1 with exactly that finding. Add the new test to the timeout group or give
it an explicit timeout. This blocks CI even though both parser modes themselves
pass locally.

## Validation performed

I built `test_tool_parser`, `test_sse` and `test_dsml_decode` from the frozen
commit in my own worktree with Debug and `EMBER_STRICT=ON`. The four corresponding
ctest entries pass, including the enabled nested-parser entry. This is evidence
that the current tests miss the integration failures, not approval of the
release candidate. The invariant check fails as described above.

The independent reproducer links the unmodified `ember_core` built from this
commit and calls the public parser, stop-boundary helper and SSE sink API. It
never sends a request or executes the named tools. With the feature unset, the
first nested fixture is rejected and no tool is emitted. With it enabled, both
counterexamples above are asserted. The owner-authored tests and the review
reproducer are separate files; no production source or existing test was edited.

From this review branch:

```sh
cmake -S . -B build-parser-review -DCMAKE_BUILD_TYPE=Debug -DEMBER_STRICT=ON
cmake --build build-parser-review --target test_tool_parser test_sse test_dsml_decode -j8
ctest --test-dir build-parser-review -R '^(tool_parser|tool_parser_nested|sse|dsml_decode)$' --output-on-failure
cc -std=c11 -Wall -Wextra -Werror -Wshadow -Wconversion -Wsign-conversion \
  -Isrc docs/reviews/evidence/nested_parser_a9f1d63.c \
  build-parser-review/libember_core.a -o build-parser-review/nested-parser-review
env -u EMBER_DSML_NESTED_VALUES build-parser-review/nested-parser-review
EMBER_DSML_NESTED_VALUES=1 build-parser-review/nested-parser-review
EMBER_DSML_NESTED_VALUES=1 build-parser-review/nested-parser-review parameter-only
python3 ci/check_invariants.py
```

The reproducer exits 0 when it observes the frozen revision's documented
counterexamples (or its expected default rejection). After a fix it should fail
those reproduction assertions; new regression tests should instead require
correct boundaries and exact argument preservation. The final invariant
command is expected to fail on this revision.

Evidence is retained under `docs/reviews/evidence/nested-parser-a9f1d63-*`:
source hashes, default/enabled outputs, the independently reachable
parameter-only case, and the invariant failure. The generation-callback impact
is established by its source use of the reproduced public helper; this review
does not claim a live server or GPU reproduction.

A revised parser/stop/emitter integration needs a fresh review before activation.
