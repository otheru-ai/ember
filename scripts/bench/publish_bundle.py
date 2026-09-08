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
import hashlib
import json
import math
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# Every file a complete bundle must carry. The harness sources are here on
# purpose: a measurement whose harness cannot be re-read is not reproducible,
# and assemble_bundle.py already copies them in.
REQUIRED = (
    "summary.json",
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


def validate(bundle: Path, release: str) -> dict:
    """Return the bundle's environment, or raise Invalid with the reason."""
    if not bundle.is_dir():
        raise Invalid(f"{bundle} is not a directory")
    missing = [n for n in REQUIRED if not (bundle / n).is_file()]
    if missing:
        raise Invalid(f"incomplete bundle, missing: {', '.join(missing)}")

    env = _read_json(bundle, "environment.json")
    summary = _read_json(bundle, "summary.json")

    # ── identity: the bundle must say what it measured ──
    runtime = env.get("runtime") or {}
    measured = runtime.get("release")
    want = release[1:] if release.startswith("v") else release
    if not measured:
        raise Invalid("environment.json runtime.release is empty; "
                      "the bundle does not name the release it measured")
    if str(measured).lstrip("v") != want:
        raise Invalid(f"bundle measured release {measured!r}, "
                      f"which is not {release!r} -- refusing to publish it "
                      f"under a version it did not measure")
    if not runtime.get("container_image"):
        raise Invalid("environment.json runtime.container_image is empty; "
                      "the measured image is unidentifiable")

    model = env.get("model") or {}
    for part in ("target", "drafter"):
        if not model.get(f"{part}_sha256"):
            raise Invalid(f"environment.json model.{part}_sha256 is missing")
        source = model.get(f"{part}_sha256_source")
        if source not in {"computed", "asserted"}:
            raise Invalid(
                f"model.{part}_sha256_source is {source!r}; it must say whether "
                f"the digest was computed or asserted")
        if source == "asserted" and not model.get(f"{part}_sha256_asserted_by"):
            raise Invalid(f"model.{part}_sha256 is asserted with no evidence "
                          f"reference; an unsourced constant is not provenance")

    # ── completeness: every summarised group needs raw rows behind it ──
    rows = [json.loads(l) for l in
            (bundle / "raw-results.jsonl").read_text().splitlines() if l.strip()]
    if not rows:
        raise Invalid("raw-results.jsonl is empty")
    # Check the groups assemble_bundle.py actually produces, and check the
    # SAMPLE COUNTS rather than mere presence. An earlier draft of this looked
    # for a "by_group" key that summarise_groups never writes, so the loop ran
    # zero times and passed every bundle -- a check that cannot fail is worse
    # than no check, because it reads as coverage.
    decode = summary.get("decode") or {}
    if decode:
        have = sum(1 for r in rows
                   if r.get("group") == "decode-256" and r.get("ok")
                   and r.get("decode_tokens_per_second"))
        if have != decode.get("samples"):
            raise Invalid(
                f"summary.decode reports {decode.get('samples')} samples but "
                f"raw-results.jsonl holds {have} usable decode-256 rows; the "
                f"aggregate does not match what is behind it")
    for name, group in (summary.get("prefill") or {}).items():
        have = sum(1 for r in rows
                   if r.get("group") == name
                   and r.get("prefill_tokens_per_second")
                   and (r.get("evaluated_prefill_tokens") or 0) > 0)
        if have != group.get("samples"):
            raise Invalid(
                f"summary.prefill[{name!r}] reports {group.get('samples')} "
                f"samples but raw-results.jsonl holds {have}")
    if not decode and not (summary.get("prefill") or {}):
        raise Invalid("summary.json reports neither decode nor prefill results")
    for name in ("workloads-spec-on.jsonl", "workloads-spec-off.jsonl"):
        if not [l for l in (bundle / name).read_text().splitlines() if l.strip()]:
            raise Invalid(f"{name} is empty")

    # ── finiteness ──
    _finite(summary, "summary.json")
    return env


def archive(bundle: Path, out_dir: Path, release: str) -> tuple[Path, Path]:
    """Write a deterministic .tar.gz and its .sha256 next to each other."""
    tag = release.lstrip("v")
    out_dir.mkdir(parents=True, exist_ok=True)
    tarball = out_dir / f"ember-{tag}-perf-bundle.tar.gz"
    members = sorted(p for p in bundle.rglob("*") if p.is_file())

    def reset(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = EPOCH
        # Directory bits vary between machines; file bits do not matter to a
        # reader and would otherwise leak the builder's umask into the digest.
        info.mode = 0o644
        return info

    # mtime=0 in the gzip header too: GzipFile stamps the current time by
    # default, which would defeat the whole point of a stable digest.
    with tarfile.open(tarball, "w:gz", compresslevel=9,
                      format=tarfile.PAX_FORMAT) as tar:
        tar.gzip_mtime = EPOCH  # documented no-op on older Pythons; see below
        for path in members:
            tar.add(path, arcname=f"ember-{tag}-perf-bundle/"
                                  f"{path.relative_to(bundle)}", filter=reset)
    # Older tarfile does not honour gzip_mtime, so normalise the 4-byte MTIME
    # field in the gzip header directly rather than trusting the attribute.
    raw = bytearray(tarball.read_bytes())
    raw[4:8] = (EPOCH).to_bytes(4, "little")
    tarball.write_bytes(bytes(raw))

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


def already_published(release, repo, tarball, work: Path) -> bool:
    """True if this exact content is already there; raise if a different one is."""
    assets = existing_assets(release, repo)
    if tarball.name not in assets:
        return False
    landed = work / ("landed-" + tarball.name)
    subprocess.run(["gh", "release", "download", release, "--repo", repo,
                    "--pattern", tarball.name, "--output", str(landed),
                    "--clobber"], check=True, capture_output=True)
    if hashlib.sha256(landed.read_bytes()).hexdigest() == \
       hashlib.sha256(tarball.read_bytes()).hexdigest():
        return True
    raise Invalid(
        f"{release} already carries a DIFFERENT {tarball.name}. Refusing to "
        f"replace a published measurement someone may have cited. Remove it "
        f"deliberately, or publish under a new name, if that is really intended.")


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
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and build the archive, upload nothing, and "
                         "print the digest")
    args = ap.parse_args()

    try:
        env = validate(args.bundle, args.release)
    except Invalid as exc:
        print(f"refusing to publish {args.bundle}: {exc}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        tarball, checksum = archive(args.bundle, work, args.release)
        digest = checksum.read_text().split()[0]
        print(f"{tarball.name}  sha256 {digest}")
        print(f"measured release {env['runtime']['release']} "
              f"image {env['runtime']['container_image']}")
        if args.dry_run:
            print("dry run: nothing uploaded")
            return 0
        try:
            if already_published(args.release, args.repo, tarball, work):
                print(f"{args.release} already carries this exact bundle; "
                      f"nothing to do")
                return 0
        except Invalid as exc:
            print(str(exc), file=sys.stderr)
            return 1
        # No --clobber: existing assets and the release notes are left alone.
        subprocess.run(["gh", "release", "upload", args.release,
                        str(tarball), str(checksum), "--repo", args.repo],
                       check=True)
        print(f"uploaded to {args.release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
