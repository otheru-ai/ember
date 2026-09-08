#!/usr/bin/env python3
"""Publish a COMPLETE benchmark bundle to an existing GitHub release, durably.

    publish_bundle.py --bundle benchmarks/ember-2026-09-08 --release v2026.9.8
    publish_bundle.py --bundle <dir> --release <tag> --dry-run

A workflow artifact expires in 90 days; a release asset does not. Certification
uploads the former and nothing has ever uploaded the latter, so every Ember
release carries zero assets. This closes that gap for bundles that already
exist -- it measures nothing and starts no server.

WHAT IT REFUSES TO PUBLISH. A bundle is a provenance claim, so an incomplete or
unverifiable one is worse than none: it looks authoritative and cannot be
checked later. Validation is therefore fail-closed on identity (the bundle must
name the release, image, model digests and harness it came from), on
completeness (every group summarised must have raw rows behind it), and on
finiteness (a NaN in a median is a broken measurement, not a small one).

NO-CLOBBER. Re-running with the same bundle is a no-op; re-running with
DIFFERENT content under an existing asset name is refused, never overwritten. A
published measurement someone may have cited does not get silently replaced.

Requires the `gh` CLI, already used by this repo's workflows. No new
dependencies, no credentials of its own -- gh supplies the auth.
"""
import argparse
import gzip
import hashlib
import io as _io
import json
import math
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the assembler's own validators and summarisers rather than restating
# them: a second implementation of "is this bundle sound" drifts from the one
# that produced the bundle, and then disagrees with it silently.
from assemble_bundle import (  # noqa: E402
    jsonl, summarise_context, summarise_groups, summarise_workloads,
    validate_workload_rows)

# Bumped whenever the rules below get stricter, and recorded in the sidecar so a
# published bundle says which contract it passed.
VALIDATOR_VERSION = 3
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
# The context sweep these releases are measured with. Depths are the sweep's
# REQUESTED targets, not the measured prompt lengths, which drift at depth.
REQUIRED_DEPTHS = (0, 1024, 4096, 16384, 32768, 65536, 98304)
# benchmark.py's throughput suite. A bundle missing one of these is a partial
# run, and a partial run published as a release baseline is a false comparison.
REQUIRED_GROUPS = ("decode-256", "prefill-128", "prefill-512", "prefill-2048",
                   "prefill-8192", "prefill-16384", "prefill-32768")
DEFAULT_WORKLOADS = 10
# registry/name:tag or registry/name@sha256:...
IMAGE_RE = re.compile(r"\A[a-z0-9.\-_/]+(:[\w.\-]+)?(@sha256:[0-9a-f]{64})?\Z")

# Every file a complete bundle must carry. The harness sources are here on
# purpose: a measurement whose harness cannot be re-read is not reproducible,
# and assemble_bundle.py already copies them in.
REQUIRED = (
    "summary.json",
    "context-sweep.jsonl",
    "environment.json",
    "host.json",
    "README.md",
    "raw-results.jsonl",
    "workloads-spec-on.jsonl",
    "workloads-spec-off.jsonl",
    "benchmark.py",
    "accept_sweep.py",
    "sweep_probe.py",
)

# Fixed so the archive of a given bundle is byte-identical on every machine and
# every run. Without this the checksum would change with the clock and the
# no-clobber comparison below could never say "same content".
EPOCH = 0


class Invalid(Exception):
    """A bundle that must not be published, with the reason a human needs."""


def _finite(value, where):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise Invalid(f"{where} is {value!r}, not a finite measurement")
    elif isinstance(value, dict):
        for k, v in value.items():
            _finite(v, f"{where}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _finite(v, f"{where}[{i}]")


def _read_json(bundle, name):
    try:
        return json.loads((bundle / name).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Invalid(f"{name} is missing or not valid JSON: {exc}") from exc


def _from_inspect(path, image, pinned):
    """Digest and source commit from a docker inspect captured at run time.

    Accepts the raw `docker inspect` array so the operator records evidence
    rather than transcribing a digest by hand. Matches the entry by repository,
    since one file can carry both releases' images.
    """
    try:
        entries = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Invalid(f"--image-inspect {path}: {exc}") from exc
    if isinstance(entries, dict):
        entries = [entries]
    repo = str(image).split("@")[0].split(":")[0]
    version = str(image).split("@")[0].rsplit(":", 1)
    want_version = version[1] if len(version) == 2 else None
    for entry in entries:
        labels = ((entry.get("Config") or {}).get("Labels") or {})
        digests = [d for d in (entry.get("RepoDigests") or [])
                   if d.split("@")[0] == repo]
        if want_version and labels.get("org.opencontainers.image.version") \
                not in (None, want_version):
            continue
        if not digests:
            continue
        found = digests[0].split("@sha256:")[-1]
        if not SHA256_RE.match(found):
            raise Invalid(f"--image-inspect RepoDigest {digests[0]!r} is not a "
                          f"SHA-256")
        if pinned and pinned != found:
            raise Invalid(f"the captured inspect digest {found} contradicts the "
                          f"digest pinned in the bundle ({pinned})")
        revision = labels.get("org.opencontainers.image.revision")
        if revision and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise Invalid(f"image revision label {revision!r} is not a commit")
        return found, revision
    raise Invalid(f"--image-inspect carries no entry for {repo} with a "
                  f"RepoDigest; a locally built image has none, and an image "
                  f"with no registry identity cannot be published as evidence")


def _digest(model, part):
    value = model.get(f"{part}_sha256")
    if not value or not SHA256_RE.match(str(value)):
        raise Invalid(f"model.{part}_sha256 is {value!r}; a bundle must carry a "
                      f"full lowercase hex SHA-256, not a placeholder")
    source = model.get(f"{part}_sha256_source")
    if source not in {"computed", "asserted"}:
        raise Invalid(f"model.{part}_sha256_source is {source!r}; it must say "
                      f"whether the digest was computed or asserted")
    if source == "asserted" and not model.get(f"{part}_sha256_asserted_by"):
        raise Invalid(f"model.{part}_sha256 is asserted with no evidence "
                      f"reference; an unsourced constant is not provenance")


def validate(bundle: Path, release: str, inspect_path=None,
             expected_workloads=None, expected_commit=None) -> dict:
    """Return the bundle's environment, or raise Invalid with the reason."""
    if not bundle.is_dir():
        raise Invalid(f"{bundle} is not a directory")
    missing = [n for n in REQUIRED if not (bundle / n).is_file()]
    if missing:
        raise Invalid(f"incomplete bundle, missing: {', '.join(missing)}")

    env = _read_json(bundle, "environment.json")
    summary = _read_json(bundle, "summary.json")

    # ── identity ──
    runtime = env.get("runtime") or {}
    measured = runtime.get("release")
    want = release[1:] if release.startswith("v") else release
    if not measured:
        raise Invalid("environment.json runtime.release is empty; "
                      "the bundle does not name the release it measured")
    if str(measured).lstrip("v") != want:
        raise Invalid(f"bundle measured release {measured!r}, which is not "
                      f"{release!r} -- refusing to publish it under a version "
                      f"it did not measure")
    image = runtime.get("container_image")
    if not image or not IMAGE_RE.match(str(image)):
        raise Invalid(f"runtime.container_image is {image!r}; the measured "
                      f"image must be a well-formed reference")
    # A tag is mutable, so a tag alone does not identify what ran. The identity
    # comes from a `docker inspect` captured at measurement time -- not from
    # resolving the tag afterwards, which reports what it points at NOW and is
    # the very mutability this check exists to close.
    pinned = str(image).split("@sha256:")[1] if "@sha256:" in str(image) else None
    revision = None
    if inspect_path:
        pinned, revision = _from_inspect(inspect_path, image, pinned)
    if not pinned:
        raise Invalid(
            f"runtime.container_image ({image}) names a mutable tag with no "
            f"digest, so it does not identify what was measured. Pass "
            f"--image-inspect with the docker inspect captured at measurement "
            f"time; its identity is recorded, never synthesised.")
    if expected_commit:
        if not revision:
            raise Invalid("--expected-commit needs an --image-inspect carrying "
                          "org.opencontainers.image.revision; there is nothing "
                          "to bind the release commit to")
        if revision != expected_commit:
            raise Invalid(
                f"the measured image was built from {revision}, but "
                f"{release} is {expected_commit}. Publishing this would "
                f"attribute one commit's performance to another.")

    model = env.get("model") or {}
    _digest(model, "target")
    _digest(model, "drafter")
    if model.get("mmproj"):
        _digest(model, "mmproj")

    # ── the measurements themselves ──
    rows = jsonl(bundle / "raw-results.jsonl")
    wl_on = jsonl(bundle / "workloads-spec-on.jsonl")
    wl_off = jsonl(bundle / "workloads-spec-off.jsonl")
    ctx = jsonl(bundle / "context-sweep.jsonl")
    if not rows:
        raise Invalid("raw-results.jsonl is empty")
    if not wl_on or not wl_off:
        raise Invalid("a workload sweep arm is empty")
    if not ctx:
        raise Invalid("context-sweep.jsonl is empty")

    # Non-finite values in the RAW rows, not only in the aggregate: a NaN that
    # is averaged away still means the run was broken.
    _finite([r for r in rows if isinstance(r, dict)], "raw-results.jsonl")
    _finite(wl_on, "workloads-spec-on.jsonl")
    _finite(wl_off, "workloads-spec-off.jsonl")
    _finite(ctx, "context-sweep.jsonl")

    # The assembler's own row validator: unique labels, no error rows, usable
    # speculative evidence, identical workload identities across both arms, and
    # a spec_cycles counter that is not inert. The expected count is DECLARED,
    # never derived from the file: deriving it from what is present cannot
    # notice a sweep that lost rows before assembly.
    try:
        validate_workload_rows(wl_on, wl_off,
                               expected_workloads or DEFAULT_WORKLOADS)
    except SystemExit as exc:
        raise Invalid(f"workload sweep: {exc}") from exc

    # ── the throughput suite must be complete, and match its own record ──
    requests = [r for r in rows if r.get("kind") in (None, "request")]
    seen = {r.get("group") for r in requests}
    absent = [g for g in REQUIRED_GROUPS if g not in seen]
    if absent:
        raise Invalid(f"raw-results.jsonl is missing the throughput group(s) "
                      f"{', '.join(absent)}; this is a partial run, and a "
                      f"partial run published as a baseline is a false "
                      f"comparison")
    summary_rows = [r for r in rows if r.get("kind") == "summary"]
    if not summary_rows:
        raise Invalid("raw-results.jsonl carries no summary row")
    declared = summary_rows[-1].get("groups") or {}
    if not declared:
        raise Invalid("the summary row declares no per-group results")
    undeclared = [g for g in REQUIRED_GROUPS if g not in declared]
    if undeclared:
        raise Invalid(f"the summary row declares no result for "
                      f"{', '.join(undeclared)}; iterating only the entries "
                      f"that exist would let a dropped group pass unnoticed")
    for name, block in declared.items():
        counted = sum(1 for r in requests if r.get("group") == name)
        if counted != block.get("samples"):
            raise Invalid(
                f"group {name} declares {block.get('samples')} samples but "
                f"raw-results.jsonl holds {counted} request rows")

    # ── vision, recomputed from the raw requests ──
    # summary.get("vision") being truthy proves only that a key exists.
    vision_rows = [r for r in requests if r.get("group") == "vision"]
    declared_vision = (summary_rows[-1].get("vision") or {})
    if not vision_rows or not declared_vision:
        raise Invalid("no vision requests in raw-results.jsonl; these releases "
                      "ship a vision tower and a bundle without it is not a "
                      "complete measurement of them")
    if len(vision_rows) != declared_vision.get("samples"):
        raise Invalid(
            f"vision declares {declared_vision.get('samples')} samples but "
            f"raw-results.jsonl holds {len(vision_rows)} vision requests")
    vision_summary = [r for r in rows if r.get("kind") == "vision_summary"]
    if not vision_summary:
        raise Invalid("raw-results.jsonl carries no vision_summary row")
    # assemble_bundle copies the raw declaration into summary.json verbatim, so
    # anything else there was edited afterwards. Counting samples alone left
    # every other vision figure -- warm_decode_tps included -- free to be
    # anything at all.
    if summary.get("vision") != declared_vision:
        raise Invalid("summary.json vision does not match the vision block in "
                      "raw-results.jsonl; it was not derived from the "
                      "measurement")
    from_row = {k: v for k, v in vision_summary[-1].items() if k != "kind"}
    if declared_vision != from_row:
        raise Invalid("the summary row's vision block does not match the "
                      "vision_summary row it should have been built from")

    # ── the context sweep must be the whole sweep, in both arms ──
    for arm in ("spec-on", "spec-off"):
        arm_rows = [r for r in ctx if r.get("config") == arm]
        if not arm_rows:
            raise Invalid(f"context-sweep.jsonl has no {arm} rows")
        bad = [r for r in arm_rows if r.get("error")
               or not isinstance(r.get("prefill_tps"), (int, float))
               or isinstance(r.get("prefill_tps"), bool)
               or not isinstance(r.get("decode_tps"), (int, float))
               or isinstance(r.get("decode_tps"), bool)
               or not math.isfinite(float(r.get("prefill_tps") or 0))
               or not math.isfinite(float(r.get("decode_tps") or 0))
               or float(r.get("prefill_tps") or 0) <= 0
               or float(r.get("decode_tps") or 0) <= 0]
        if bad:
            raise Invalid(f"context sweep {arm} has {len(bad)} row(s) with an "
                          f"error or a non-positive timing; first target="
                          f"{bad[0].get('target')}")
        targets = [r.get("target") for r in arm_rows]
        if len(targets) != len(set(targets)):
            raise Invalid(f"context sweep {arm} repeats a depth")
        missing = [d for d in REQUIRED_DEPTHS if d not in set(targets)]
        if missing:
            raise Invalid(
                f"context sweep {arm} is missing depth(s) "
                f"{', '.join(str(d) for d in missing)}; the published curve "
                f"must cover the whole suite, not the part that succeeded")
    depths = summarise_context(ctx)

    # ── aggregates must be the ones these rows produce ──
    recomputed = summarise_groups(rows)
    if not recomputed.get("decode"):
        raise Invalid("no usable decode-256 rows in raw-results.jsonl")
    if not recomputed.get("prefill"):
        raise Invalid("no usable prefill-* rows in raw-results.jsonl")
    for key in ("decode", "prefill"):
        if summary.get(key) != recomputed[key]:
            raise Invalid(
                f"summary.json {key} does not match what raw-results.jsonl "
                f"produces; the aggregate was not derived from these rows")
    if summary.get("by_workload") != summarise_workloads(wl_on, wl_off):
        raise Invalid("summary.json by_workload does not match the workload "
                      "sweep files")
    if summary.get("by_context_depth") != depths:
        raise Invalid("summary.json by_context_depth does not match "
                      "context-sweep.jsonl")
    _finite(summary, "summary.json")

    env["_publication"] = {
        "image_digest": pinned,
        "image_revision": revision,
        "release": release,
        "validator_version": VALIDATOR_VERSION,
        "workloads_expected": expected_workloads or DEFAULT_WORKLOADS,
    }
    return env


def archive(bundle: Path, out_dir: Path, release: str, sidecar=None) -> tuple:
    """Write a deterministic .tar.gz and its .sha256 next to each other."""
    tag = release.lstrip("v")
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() == bundle.resolve() or \
            bundle.resolve() in out_dir.resolve().parents:
        raise Invalid("the archive must be written outside the bundle, or it "
                      "would try to contain itself")
    tarball = out_dir / f"ember-{tag}-perf-bundle.tar.gz"

    members = sorted(p for p in bundle.rglob("*"))
    for path in members:
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise Invalid(f"{path.relative_to(bundle)} is a symlink or a "
                          f"special file; a published bundle carries only "
                          f"regular files")
    files = [p for p in members if p.is_file()]

    def reset(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = EPOCH
        # Directory bits vary between machines and file bits do not matter to a
        # reader; pinning them keeps the builder's umask out of the digest.
        info.mode = 0o644
        return info

    # GzipFile(mtime=0, filename="") is the supported way to get a stable gzip
    # header. An earlier draft set a `tar.gzip_mtime` attribute that tarfile
    # does not have -- it silently did nothing -- and then rewrote the header
    # bytes by hand. Both are gone.
    with open(tarball, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9,
                           fileobj=raw, mtime=EPOCH) as gz:
            with tarfile.open(fileobj=gz, mode="w",
                              format=tarfile.PAX_FORMAT) as tar:
                for path in files:
                    tar.add(path, arcname=f"ember-{tag}-perf-bundle/"
                                          f"{path.relative_to(bundle)}",
                            filter=reset)
                if sidecar is not None:
                    # Written into the ARCHIVE, never into the operator's
                    # bundle: the captured identity and the contract version
                    # must travel with what is published, and the raw
                    # measurement directory stays exactly as it was measured.
                    blob = (json.dumps(sidecar, indent=2, sort_keys=True)
                            + "\n").encode()
                    info = tarfile.TarInfo(
                        f"ember-{tag}-perf-bundle/publication.json")
                    info.size = len(blob)
                    tar.addfile(reset(info), _io.BytesIO(blob))

    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    checksum = out_dir / (tarball.name + ".sha256")
    checksum.write_text(f"{digest}  {tarball.name}\n")
    return tarball, checksum


def gh_json(args, repo):
    out = subprocess.run(["gh", *args, "--repo", repo],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out or "{}")


def existing_assets(release, repo):
    data = gh_json(["release", "view", release, "--json", "assets"], repo)
    return {a["name"]: a for a in data.get("assets") or []}


def release_commit(release, repo):
    """The commit the release tag points at, resolved online."""
    out = subprocess.run(["gh", "api", f"repos/{repo}/commits/{release}",
                          "--jq", ".sha"], capture_output=True, text=True)
    if out.returncode != 0:
        raise Invalid(f"could not resolve {release} to a commit: "
                      f"{out.stderr.strip()[:200]}")
    sha = out.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise Invalid(f"{release} resolved to {sha!r}, not a commit")
    return sha


def _downloaded(release, repo, name, work: Path):
    dest = work / ("landed-" + name)
    subprocess.run(["gh", "release", "download", release, "--repo", repo,
                    "--pattern", name, "--output", str(dest), "--clobber"],
                   check=True, capture_output=True)
    return dest


def publication_state(release, repo, tarball, checksum, work: Path):
    """One of: 'absent', 'complete', or a list of the assets still to upload.

    Checking only the archive was wrong: an upload interrupted between the two
    assets leaves the archive present and the checksum missing, and reporting
    that as complete would strand it permanently.
    """
    assets = existing_assets(release, repo)
    local = {p.name: p for p in (tarball, checksum)}
    present = [n for n in local if n in assets]
    if not present:
        return "absent"
    for name in present:
        landed = _downloaded(release, repo, name, work)
        if hashlib.sha256(landed.read_bytes()).hexdigest() != \
           hashlib.sha256(local[name].read_bytes()).hexdigest():
            raise Invalid(
                f"{release} already carries a DIFFERENT {name}. Refusing to "
                f"replace a published measurement someone may have cited. "
                f"Remove it deliberately if that is really intended.")
    outstanding = [local[n] for n in local if n not in assets]
    return "complete" if not outstanding else outstanding


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True,
                    help="a benchmarks/ember-<date> directory from "
                         "scripts/benchmark_bundle.sh")
    ap.add_argument("--release", required=True,
                    help="existing release tag, e.g. v2026.9.8. It must already "
                         "exist; this never creates a release or edits its notes")
    ap.add_argument("--repo", default="otheru-ai/ember",
                    help="owner/name (default otheru-ai/ember)")
    ap.add_argument("--image-inspect", type=Path, default=None,
                    help="`docker inspect` output captured at measurement time, "
                         "when the bundle recorded only a mutable tag. Its "
                         "RepoDigest and image.revision label are recorded; "
                         "nothing is synthesised and the tag is never resolved "
                         "after the fact")
    ap.add_argument("--expected-commit", default=None,
                    help="bind the measured image to this commit offline. "
                         "Online, the release tag is resolved with gh and used "
                         "instead, so this is for --dry-run and for hosts with "
                         "no network")
    ap.add_argument("--expected-workloads", type=int, default=DEFAULT_WORKLOADS,
                    help=f"how many workloads the sweep must carry "
                         f"(default {DEFAULT_WORKLOADS})")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and build the archive, upload nothing, and "
                         "print the digest")
    args = ap.parse_args()

    expected = args.expected_commit
    if not args.dry_run:
        # The release is the authority for what commit this version is, so the
        # resolution ALWAYS happens online. --expected-commit may only agree
        # with it; letting it substitute would have made the binding
        # operator-assertable, which is the opposite of the point.
        try:
            resolved = release_commit(args.release, args.repo)
        except Invalid as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if expected and expected != resolved:
            print(f"--expected-commit {expected} disagrees with {args.release}, "
                  f"which is {resolved}", file=sys.stderr)
            return 1
        expected = resolved

    try:
        env = validate(args.bundle, args.release, args.image_inspect,
                       args.expected_workloads, expected)
    except Invalid as exc:
        print(f"refusing to publish {args.bundle}: {exc}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        pub = env["_publication"]
        try:
            tarball, checksum = archive(args.bundle, work / "out",
                                        args.release, sidecar=pub)
        except Invalid as exc:
            print(f"refusing to publish {args.bundle}: {exc}", file=sys.stderr)
            return 1
        digest = checksum.read_text().split()[0]
        print(f"{tarball.name}  sha256 {digest}")
        print(f"measured release {env['runtime']['release']}  "
              f"image sha256:{pub['image_digest']}  "
              f"revision {pub['image_revision']}  "
              f"validator v{pub['validator_version']}")
        if args.dry_run:
            print("dry run: nothing uploaded")
            return 0
        try:
            state = publication_state(args.release, args.repo, tarball,
                                      checksum, work)
        except Invalid as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if state == "complete":
            print(f"{args.release} already carries this exact bundle; "
                  f"nothing to do")
            return 0
        upload = [tarball, checksum] if state == "absent" else state
        if state != "absent":
            print("completing a partial upload: "
                  + ", ".join(p.name for p in upload))
        # No --clobber: existing assets and the release notes are left alone.
        subprocess.run(["gh", "release", "upload", args.release,
                        *[str(p) for p in upload], "--repo", args.repo],
                       check=True)
        print(f"uploaded to {args.release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
