# Contributing

Contributions are welcome. Before opening a pull request:

1. Keep the backend ABI implemented by both the stub and real backends.
2. Preserve the two explicit CMake source lists when adding `src/` files.
3. Run `python3 ci/check_invariants.py`.
4. Build with `EMBER_STRICT=ON` and run the full GPU-free test suite.
5. Add focused regression coverage for behavior changes.

```bash
cmake -S . -B build -DEMBER_STRICT=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

ROCm changes should also be built on gfx1151 hardware and checked with
`ember-dflash --validate-prompt`. Do not run GPU tests against a shared device
without coordinating the maintenance window.

XDNA changes must preserve optional fallback and the versioned provider ABI,
update `providers/xdna2/VENDOR.md` when a pinned source moves, and pass the
packaged kernel validators. Performance or promotion changes also require the
trained two-session differential, output-hash corpus, and aggregate-throughput
gate documented in `docs/xdna2-moe-prototype.md`. An NPU microbenchmark alone
is not a serving result.

Use concise commits in the form `type(scope): subject`. By contributing, you
agree that your contribution is licensed under the repository's MIT license.

## Release versions

Ember uses calendar versions in `YEAR.MONTH.DAY` form without zero-padding.
Do not update `VERSION` or create a release tag manually. Once an immutable
candidate passes the protected gfx1151 certification workflow, release
automation groups the conventional commits into `CHANGELOG.md`, updates the
version and Compose image pin in a metadata-only child commit, and creates
the matching annotated `vYEAR.MONTH.DAY` tag. The container workflow rejects a
tag unless that commit's parent is the certified executable tree and only the
three approved metadata files changed.

Only one release is published per calendar day. If another change is needed on
the same day, keep it on `main` and include it in the next dated release.
