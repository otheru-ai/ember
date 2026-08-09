# Promoting development changes

`ember-dev` and `ember` have unrelated Git histories by design. Promotion is a
reviewed content merge, not a repository merge, cherry-pick, rsync, or copy.
The release repository remains the authority for packaging, security policy,
sanitized documentation, CI, and container behavior.

## Ownership boundary

Automatically mergeable paths are implementation-oriented:

- `CMakeLists.txt`
- `src/`, `engine/`, `vendor/`, `share/`, and `tools/`
- source and test-data files under `test/`

Root documentation, `docs/`, `scripts/`, deployment files, and release policy
require an explicit include-or-skip decision. Generated reports, captured live
requests, production-derived fixtures, bytecode, and `.env` are blocked from
automatic inclusion. `engine/VENDOR.md` is manual because release provenance
must describe the exact snapshot and local modifications accurately.

## Promotion cycle

1. Finish and commit the development work. Record GPU results in development;
   do not copy raw operational reports into the release tree.
2. Make the release worktree clean. Release-only fixes should be committed
   before promotion so the three-way merge can preserve them.
3. Inspect the pending range:

   ```bash
   scripts/promote_from_dev.py
   ```

   The base comes from `.release/ember-dev-base`; the target defaults to the
   local sibling checkout's `HEAD`. Use `--dev PATH` and `--target COMMIT` when
   promoting from another checkout or a deliberately older tested commit.
4. Review every development commit and every changed path. Resolve manual paths
   explicitly:

   ```bash
   scripts/promote_from_dev.py \
     --include docs/continuous-batching.md \
     --skip reports/internal-validation.json
   ```

   `--include` means the normal three-way content merge is appropriate.
   `--skip` means the path was intentionally omitted or was separately adapted
   and committed in release form. Blocked paths can only be skipped.
5. A reported conflict is never overwritten. Port that path manually, test and
   commit the resolution, then acknowledge it with `--skip PATH` in the final
   promotion command.
6. Apply only after the plan is fully resolved:

   ```bash
   scripts/promote_from_dev.py \
     --include docs/continuous-batching.md \
     --skip reports/internal-validation.json \
     --apply
   ```

   Apply mode requires both repositories to be clean. It computes every merge
   before writing, updates the base SHA and promotion ledger, but never stages,
   commits, fetches, or pushes.
7. Review `git diff`, run the invariant checker, strict Debug and Release suites,
   sanitizer/analyzer jobs, and the ROCm validator when backend code changed.
8. Commit the result as one release-side promotion, for example:

   ```text
   feat(release): promote ember-dev d1571a0
   ```

   This records provenance without importing development commit history.

## Feedback to development

Functional fixes discovered while hardening `ember` must also be reapplied and
tested in `ember-dev`; otherwise development will repeatedly reintroduce a bug
that the release merge has to preserve downstream. Packaging, policy, release
documentation, and container-only changes remain release-owned.

Never advance `.release/ember-dev-base` by hand. The SHA means every development
change through that commit was either included or explicitly accounted for in
`.release/promotion-log.md`.
