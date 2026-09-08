# Publishing a benchmark bundle

A workflow artifact expires in 90 days. A release asset does not. Certification
uploads the former (`gfx1151-certify.yml`, `actions/upload-artifact`,
`retention-days: 90`) and nothing has ever uploaded the latter, which is why
every Ember release carries zero assets and why `docs/perf/data.json` still ends
at 2026.9.3.

`scripts/bench/publish_bundle.py` closes that gap for bundles that already
exist. It measures nothing, starts no server, and needs no GPU.

## Publish one bundle

```sh
# Validate and build the archive without uploading. Always do this first.
python3 scripts/bench/publish_bundle.py \
  --bundle benchmarks/ember-2026-09-08 --release v2026.9.8 --dry-run

# Upload to the existing release.
python3 scripts/bench/publish_bundle.py \
  --bundle benchmarks/ember-2026-09-08 --release v2026.9.8
```

It uploads two assets: `ember-<version>-perf-bundle.tar.gz` and its `.sha256`.
The release's tag, notes and any existing assets are untouched — there is no
`--clobber` anywhere in this path, and the tool never creates or edits a
release.

## What it refuses, and why

A bundle is a provenance claim. An incomplete one is worse than none, because it
looks authoritative and cannot be checked afterwards. Validation is fail-closed:

- **Identity.** `environment.json` must name the release it measured, and that
  must match the tag being published — a bundle is never published under a
  version it did not measure. It must also carry the container image and a
  SHA-256 for the target and drafter, each marked `computed` or `asserted`, with
  an evidence reference when asserted.
- **Completeness.** Every required file must be present, including the three
  harness sources: a measurement whose harness cannot be re-read is not
  reproducible. Aggregates are checked against the rows behind them by SAMPLE
  COUNT, not by presence — `summary.decode.samples` must equal the usable
  `decode-256` rows in `raw-results.jsonl`, and the same for every
  `prefill-*` group.
- **Finiteness.** A `NaN` or infinity anywhere in `summary.json` is a broken
  measurement, not a small one.

## Re-running is safe; overwriting is not

The archive is byte-identical for identical bundle content: member mtimes, uid,
gid and mode are normalised and the gzip header timestamp is zeroed. So the tool
can compare what it built against what is already published.

- Same content already on the release → reports it and exits 0.
- **Different** content under the same asset name → **refuses**. A published
  measurement someone may have cited is never silently replaced. Remove it
  deliberately if that is genuinely intended.

## Updating the perf site from both bundles

Asset publication alone does not move the chart; `docs/perf/data.json` is what
the page renders. Pass both measured bundles in one run, merging into the
existing file so the five already-published releases survive:

```sh
python3 scripts/bench/build_perf_site_data.py \
  --bundle benchmarks/<the v2026.9.5 bundle> \
  --bundle benchmarks/<the v2026.9.8 bundle> \
  --merge-into docs/perf/data.json \
  --certified \
  --id <bundle-id>=2026.9.5 \
  --id <bundle-id>=2026.9.8 \
  --out docs/perf/data.json
```

`--merge-into` is read completely before `--out` is written, so pointing both at
`docs/perf/data.json` is safe and is the intended in-place form. Without
`--merge-into` the output contains only the bundles passed on that command line
and the earlier releases are lost.

`--id OLD=NEW` renames a bundle whose id is a commit SHA to the release version;
omit it when the bundle already carries the version. `--certified` marks these
as coming from release certification rather than a manual run. The merge line
prints how many releases were kept, added and replaced — read it before
committing the result.

## Requirements

The `gh` CLI, already used by this repo's workflows, supplying its own auth.
No other dependencies and no credentials of the tool's own.
