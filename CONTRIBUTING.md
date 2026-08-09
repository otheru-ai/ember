# Contributing

Contributions are welcome. Before opening a pull request:

1. Keep the backend ABI implemented by both the stub and GPU backends.
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

Use concise commits in the form `type(scope): subject`. By contributing, you
agree that your contribution is licensed under the repository's MIT license.

## Release versions

Ember uses calendar versions in `YEAR.MONTH.DAY` form without zero-padding.
Before cutting a release, update the root `VERSION` file and add the matching
dated section to `CHANGELOG.md`. Create a signed or annotated Git tag by adding
`v` to that exact value (for example, `v2026.8.9`). The container workflow
rejects release tags that do not match `VERSION`.

Only one release is published per calendar day. If another change is needed on
the same day, keep it on `main` and include it in the next dated release.
