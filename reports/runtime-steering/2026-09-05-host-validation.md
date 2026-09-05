# Directional steering: host validation, 2026-09-05 UTC

Candidate branch: `feat/ds4-runtime-steering`, based on `1e3070b`.
Built and tested by Codex; independent review and all hardware gates pending.

| Check | Result |
|---|---|
| Strict Debug host suite | 73/73 |
| Strict Release host suite | 73/73 |
| ASan + UBSan + default LeakSanitizer, non-PIE | 73/73 |
| Coverage build host suite | 73/73 |
| Coverage ratchet | Pass; new loader 96.7%, inline host projection 100% |
| GCC `-fanalyzer`, cppcheck warning/portability | Pass |
| `ci/check_invariants.py`, `git diff --check` | Pass |
| C ABI member sequence | Prior 10 members unchanged; 3 appended |
| ROCm 10 dev container, strict Release | `ember-dflash` and graph test built |
| Production projection helper on ggml CPU | Pass, widths 1/2/4/17/64, attention and FFN, scales 0/1/-1/3.5 |
| Projection removed from disposable helper copy | Test exits 1, 1086 assertions fail |

The negative control substitutes the input for the projected output while
retaining the same test and link libraries. It proves the graph test detects
a dropped projection; it does not prove the whole model reaches every graph
site. The latter requires source review and the hardware differential gate.

The initial PIE sanitizer run encountered repeated startup hangs and
`AddressSanitizer:DEADLYSIGNAL`, including unrelated tests. An empty C `main`
compiled with `-fsanitize=address,undefined` reproduced 7 timeouts in 20 runs
(0.5-second timeout). Adding `-fno-pie -no-pie` passed 20/20. The full suite
then passed with non-PIE sanitizer executables, keeping leak detection enabled.
No sanitizer was disabled. This is evidence of a local execution limitation,
not an explanation inferred from the steering test alone.

Reproduction uses the ordinary commands in `AGENTS.md`, with these local
sanitizer flags:

```sh
cmake -S . -B build-steering-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie -g' \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie -g' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined -no-pie'
cmake --build build-steering-asan -j8
ctest --test-dir build-steering-asan --output-on-failure -j8
```

The container ran with `--network none` and no device mounts. Its command was
`cmake -S . -B build-steering-rocm -DCMAKE_BUILD_TYPE=Release -DEMBER_ENGINE=ON
-DEMBER_STRICT=ON`, followed by building `ember-dflash` and
`test_ds4_steering_graph`. The latter executes the CPU backend explicitly.

Retained local evidence:
`/home/mythos/Projects/vision-release-evidence/runtime-steering-20260905/`.
This includes build and test logs, analyzer output, and the compressed original
PIE sanitizer failure log with its uncompressed SHA-256 and byte count.

No target weights were loaded and no GPU, runner, workflow or production action
was taken. Scale 3.5 arithmetic coverage is **not** the behavioral positive
control. The refusal response, repetition outcome, tool quality, GPU AR/spec
parity, native vision behavior, throughput and memory effects remain
unmeasured. See [the deployment gate](../../docs/directional-steering.md#validation-and-release-gate).
