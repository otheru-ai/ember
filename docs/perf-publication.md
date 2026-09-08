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
looks authoritative and cannot be checked afterwards. Validation reuses
`assemble_bundle.py`'s own validators and summarisers rather than restating
them — a second opinion about what makes a bundle sound drifts from the one that
produced it, and then disagrees silently.

- **Identity.** `environment.json` must name the release it measured, and that
  must match the tag being published. Target, drafter and (when present) mmproj
  digests must be full lowercase SHA-256, each marked `computed` or `asserted`
  with an evidence reference when asserted. The container image must be a
  well-formed reference **pinned by digest**: a tag is mutable and does not
  identify what ran. Real bundles record only a tag, so pass
  `--image-inspect <file>` — the `docker inspect` output captured at
  measurement time. Its `RepoDigest` becomes the recorded identity and its
  `org.opencontainers.image.revision` label is bound to the release commit,
  which the tool resolves from the release itself with `gh` (offline, use
  `--expected-commit`). Resolving the tag afterwards is refused by
  construction: that reports what it points at *now*, which is the mutability
  this check exists to close. Nothing is ever synthesised, and an inspect that
  contradicts the bundle is refused.
- **Completeness.** All eleven required files, including `context-sweep.jsonl`
  and the three harness sources — a measurement whose harness cannot be re-read
  is not reproducible. The workload sweep goes through
  `validate_workload_rows`: unique labels, no error rows, usable speculative
  evidence, identical workload identities in both arms, and a `spec_cycles`
  counter that is not inert. The throughput suite must be whole — every one of
  `decode-256`, `prefill-128/512/2048/8192/16384/32768` — and each group's row
  count must equal the count its own summary record declares. The context
  sweep must carry every depth of the suite (0, 1024, 4096, 16384, 32768,
  65536, 98304) in **both** arms, with no error rows and no non-positive
  timing: a curve published with the depths that happened to succeed is a
  false comparison. Vision is counted from the raw requests and checked
  against the declared sample count, because a truthy `summary["vision"]`
  proves only that a key exists.
- **Aggregates must be the ones these rows produce.** Sample counts cannot
  catch a fabricated median, so `summary.json`'s `decode`, `prefill`,
  `by_workload` and `by_context_depth` are recomputed from the raw files with
  the assembler's own summarisers and compared.
- **Finiteness.** A `NaN` or infinity in `summary.json` *or in any raw row* is
  refused. A NaN the median averages away still means the run was broken.

The workload count is **declared, not derived** — 10 by default, overridable
with `--expected-workloads N`. Deriving it from the rows present cannot detect
a sweep that lost rows before assembly, which is exactly the case it exists to
catch.

The archive carries a generated `publication.json` recording the image digest,
the bound revision, the release and the validator version. It is written into
the archive only; the measured bundle directory is never modified.

## Re-running is safe; overwriting is not

The archive is byte-identical for identical bundle content: member mtimes, uid,
gid and mode are normalised and the gzip header timestamp is zeroed. So the tool
can compare what it built against what is already published.

- Same content already on the release → reports it and exits 0.
- Archive present but checksum missing (an upload interrupted between the two)
  → uploads only what is outstanding. Checking the archive alone reported this
  as complete and stranded the release without its checksum, permanently.
- **Different** content under either asset name → **refuses**. A published
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
