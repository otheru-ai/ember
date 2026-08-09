# Vendored: XGrammar

Constrained-decoding core used to make malformed DSML tool calls
unrepresentable at the sampler rather than only rejectable afterwards. See
`src/model/tool_grammar.h` for how the grammar is derived from a request's tool
schemas.

| | |
|---|---|
| Upstream | https://github.com/mlc-ai/xgrammar |
| Tag | `v0.2.3` |
| Commit | `557becfb64c503ae9c04344b0047661f43f44320` |
| License | Apache-2.0 (`LICENSE`, `NOTICE`) |

## Why a prebuilt library is not used

The published wheel ships `lib/libxgrammar.a`, but its members are **GCC LTO
bitcode**, not machine code — `.gnu.lto_*` sections with a 7-entry real
`.symtab`. `nm` reports ~900 symbols through the LTO plugin that have no code
behind them, so clang/lld cannot link it under any flags, and extracting the
objects fails identically. Building from source with our own toolchain is the
only option. (Mangled names matched byte-for-byte, so this was never an ABI
problem.)

## What was taken

Only what the static core needs:

- `cpp/` — the library sources, including the `cpp/support/cpptrace.h` shim
- `include/` — the public C++ API
- `3rdparty/dlpack/include` — `DLTensor`, used by the bitmask API
- `3rdparty/picojson/picojson.h` — header-only, used by the JSON-schema path

## What was omitted, and why it is safe

- **`cpp/tvm_ffi/`** — the Python bindings. Upstream's own CMakeLists already
  filters these sources out of the core library
  (`list(FILTER XGRAMMAR_SOURCES_PATH EXCLUDE REGEX ".../cpp/tvm_ffi/.*\.cc")`).
  Dropping the directory also removes the `find_package(tvm_ffi)` requirement
  that otherwise fails configuration before a single file compiles.
- **`3rdparty/googletest`** (4.4 MB) — upstream's C++ test dependency. Nothing
  in `cpp/` or `include/` references it, and ember uses a hand-rolled `CHECK`
  macro rather than a test framework.
- **`3rdparty/cpptrace`** (1.4 MB) — optional, gated by
  `XGRAMMAR_ENABLE_CPPTRACE`, which upstream defaults to `OFF`. The include in
  `cpp/support/logging.h` resolves to the in-tree shim, so with the flag off
  there is no third-party library to compile at all.
- Upstream's `CMakeLists.txt` files — replaced by ours for the reasons above.

Result: 1.2 MB and 64 files, versus 9.9 MB for the full tree.

## Porting an upstream fix

Diff `cpp/` and `include/` against the pinned commit; that is where fixes land.
Re-copy those two directories, re-check the omissions above still hold, and
update the commit and tag in this file.
