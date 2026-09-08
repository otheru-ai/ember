#!/usr/bin/env python3
"""Contract for scripts/bench/publish_bundle.py. No GPU, no network, no gh.

The gh paths are exercised against a stub subprocess.run, so idempotence,
partial-upload repair and mismatch refusal are covered without a release.
"""
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

import publish_bundle as pb  # noqa: E402
from assemble_bundle import (  # noqa: E402
    summarise_context, summarise_groups, summarise_workloads)
from publish_bundle import Invalid, REQUIRED, archive, validate  # noqa: E402

DATA = ROOT / "test" / "data" / "perf-bundle"
IMAGE = "ghcr.io/otheru-ai/ember:2026.9.8"
INSPECT = DATA / "image-inspect.json"
DIGEST = "d0a558b3836db77cdddbe00d9b19500487543f413e9c6f4c3155d0063bac6b6a"
COMMIT = "d46ecc956c545f7bc70a0a7f450330aa1e6efa93"


def rows_of_bundle(bundle, name):
    return [json.loads(l) for l in (bundle / name).read_text().splitlines()
            if l.strip()]


def rows_of(name):
    return [json.loads(l) for l in (DATA / name).read_text().splitlines()
            if l.strip()]


def make_bundle(directory: Path) -> Path:
    """A bundle that VALIDATES, so each test can break exactly one thing.

    raw-results.jsonl and both workload files are the REAL 2026.9.8 measurement
    (test/data/perf-bundle), not hand-written: a synthetic fixture would omit
    the groups the validator is supposed to require and still pass, which is
    the failure mode this whole file exists to avoid. summary.json is generated
    by the assembler's own summarisers, so the comparison stays under test
    rather than the fixture and the tool agreeing on a constant. Only the
    context sweep is constructed, because no real one was available.
    """
    bundle = directory / "ember-2026-09-08"
    bundle.mkdir(parents=True)
    for name in ("raw-results.jsonl", "workloads-spec-on.jsonl",
                 "workloads-spec-off.jsonl"):
        shutil.copy(DATA / name, bundle / name)
    rows, wl_on, wl_off = (rows_of("raw-results.jsonl"),
                           rows_of("workloads-spec-on.jsonl"),
                           rows_of("workloads-spec-off.jsonl"))

    ctx = [{"config": c, "target": t, "prompt_tokens": max(t, 1),
            "prefill_tps": 900.0 - t / 1000.0, "decode_tps": d, "accept": 0.7}
           for t in pb.REQUIRED_DEPTHS
           for c, d in (("spec-on", 40.0), ("spec-off", 20.0))]
    (bundle / "context-sweep.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ctx))

    summary = summarise_groups(rows)
    # assemble_bundle copies the raw summary row's vision block verbatim; the
    # fixture must do the same or the equality check has nothing to compare.
    summary["vision"] = [r for r in rows if r.get("kind") == "summary"][-1]["vision"]
    summary["by_workload"] = summarise_workloads(wl_on, wl_off)
    summary["by_context_depth"] = summarise_context(ctx)
    (bundle / "summary.json").write_text(json.dumps(summary))
    (bundle / "environment.json").write_text(json.dumps({
        "runtime": {"release": "2026.9.8", "container_image": IMAGE},
        "model": {"target_sha256": "a" * 64, "target_sha256_source": "computed",
                  "drafter_sha256": "b" * 64,
                  "drafter_sha256_source": "asserted",
                  "drafter_sha256_asserted_by": "gfx1151 certify, run 1"},
    }))
    for name in REQUIRED:
        if not (bundle / name).exists():
            (bundle / name).write_text("{}\n" if name.endswith(".json")
                                       else "row\n")
    return bundle


class ValidateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.bundle = make_bundle(self.dir / "b")
        self.addCleanup(self.tmp.cleanup)

    def ok(self, **kw):
        kw.setdefault("inspect_path", INSPECT)
        return validate(self.bundle, kw.pop("release", "v2026.9.8"), **kw)

    def rewrite(self, name, obj):
        (self.bundle / name).write_text(json.dumps(obj))

    def write_rows(self, name, rows):
        (self.bundle / name).write_text(
            "".join(json.dumps(r) + "\n" for r in rows))

    def summary(self):
        return json.loads((self.bundle / "summary.json").read_text())

    def environment(self):
        return json.loads((self.bundle / "environment.json").read_text())

    # ── the happy path ──
    def test_a_complete_bundle_validates(self):
        env = self.ok()
        self.assertEqual(env["_publication"]["image_digest"], DIGEST)
        self.assertEqual(env["_publication"]["image_revision"], COMMIT)

    def test_tag_may_carry_the_v_prefix_or_not(self):
        self.ok(release="2026.9.8")

    # ── identity ──
    def test_a_bundle_measured_on_another_release_is_refused(self):
        with self.assertRaisesRegex(Invalid, "did not measure"):
            self.ok(release="v2026.9.5")

    def test_a_mutable_tag_with_no_captured_inspect_is_refused(self):
        with self.assertRaisesRegex(Invalid, "mutable tag with no digest"):
            validate(self.bundle, "v2026.9.8")

    def test_the_image_must_be_built_from_the_release_commit(self):
        with self.assertRaisesRegex(Invalid, "attribute one commit"):
            self.ok(expected_commit="f" * 40)
        self.ok(expected_commit=COMMIT)

    def test_binding_a_commit_without_an_inspect_is_refused(self):
        # Reachable only when the BUNDLE pins the digest itself, so there is an
        # identity but no revision label to bind the release commit to. With a
        # tag-only image the missing-digest refusal fires first.
        env = self.environment()
        env["runtime"]["container_image"] = f"ghcr.io/otheru-ai/ember@sha256:{DIGEST}"
        self.rewrite("environment.json", env)
        with self.assertRaisesRegex(Invalid, "nothing to bind"):
            validate(self.bundle, "v2026.9.8", expected_commit=COMMIT)

    def test_an_inspect_for_another_repository_is_refused(self):
        other = self.dir / "other-inspect.json"
        entry = json.loads(INSPECT.read_text())[0]
        entry["RepoDigests"] = ["ghcr.io/someone/else@sha256:" + "9" * 64]
        other.write_text(json.dumps([entry]))
        with self.assertRaisesRegex(Invalid, "no entry for"):
            self.ok(inspect_path=other)

    def test_every_required_file_is_required(self):
        for name in REQUIRED:
            with self.subTest(name=name):
                shutil.copy(self.bundle / name, self.dir / "held")
                (self.bundle / name).unlink()
                with self.assertRaisesRegex(Invalid, "incomplete bundle"):
                    self.ok()
                shutil.copy(self.dir / "held", self.bundle / name)

    def test_model_digests_must_be_real_sha256(self):
        for value in ("not-a-digest", "A" * 64, "a" * 63):
            with self.subTest(value=value):
                env = self.environment()
                env["model"]["target_sha256"] = value
                self.rewrite("environment.json", env)
                with self.assertRaisesRegex(Invalid, "full lowercase hex"):
                    self.ok()
                self.setUp()

    def test_asserted_digest_needs_evidence(self):
        env = self.environment()
        del env["model"]["drafter_sha256_asserted_by"]
        self.rewrite("environment.json", env)
        with self.assertRaisesRegex(Invalid, "evidence reference"):
            self.ok()

    # ── the throughput suite must be complete and self-consistent ──
    def test_a_missing_throughput_group_is_refused(self):
        for group in ("decode-256", "prefill-128", "prefill-32768"):
            with self.subTest(group=group):
                rows = [r for r in rows_of("raw-results.jsonl")
                        if r.get("group") != group]
                self.write_rows("raw-results.jsonl", rows)
                with self.assertRaisesRegex(Invalid, "missing the throughput"):
                    self.ok()
                self.setUp()

    def test_a_group_count_that_disagrees_with_its_record_is_refused(self):
        kept, dropped = [], False
        for r in rows_of("raw-results.jsonl"):
            if not dropped and r.get("group") == "prefill-512":
                dropped = True
                continue
            kept.append(r)
        self.write_rows("raw-results.jsonl", kept)
        with self.assertRaisesRegex(Invalid, "declares 3 samples"):
            self.ok()

    def test_vision_is_counted_from_raw_requests_not_read_off_the_summary(self):
        kept, removed = [], False
        for r in rows_of("raw-results.jsonl"):
            if not removed and r.get("group") == "vision":
                removed = True
                continue
            kept.append(r)
        self.write_rows("raw-results.jsonl", kept)
        # The summary row still CLAIMS its original sample count, so a
        # truthiness check on summary["vision"] would pass this.
        with self.assertRaisesRegex(Invalid, "declares"):
            self.ok()

    def test_an_edited_vision_metric_is_refused(self):
        # Counting samples alone left every other vision figure free.
        summary = self.summary()
        summary["vision"]["warm_decode_tps"] = 999999
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "not derived from the measurement"):
            self.ok()

    def test_a_summary_row_vision_block_that_drifted_is_refused(self):
        # Both the summary row and summary.json are edited consistently, so the
        # only remaining disagreement is with the vision_summary row the block
        # should have been built from. Editing one alone trips the earlier
        # check and would not reach this one.
        rows = rows_of("raw-results.jsonl")
        drifted = None
        for r in rows:
            if r.get("kind") == "summary":
                drifted = dict(r["vision"], warm_decode_tps=123.0)
                r["vision"] = drifted
        self.write_rows("raw-results.jsonl", rows)
        summary = self.summary()
        summary["vision"] = drifted
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "vision_summary row"):
            self.ok()

    def test_an_undeclared_required_group_is_refused(self):
        rows = rows_of("raw-results.jsonl")
        for r in rows:
            if r.get("kind") == "summary":
                r["groups"].pop("prefill-2048")
        self.write_rows("raw-results.jsonl", rows)
        with self.assertRaisesRegex(Invalid, "declares no result for"):
            self.ok()

    # ── aggregates ──
    def test_an_invented_aggregate_with_matching_counts_is_refused(self):
        summary = self.summary()
        summary["decode"]["median_tps"] = 99.0
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "not derived from these rows"):
            self.ok()

    def test_an_invented_workload_table_is_refused(self):
        summary = self.summary()
        summary["by_workload"]["code"]["speedup"] = 9.9
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "by_workload"):
            self.ok()

    def test_an_invented_depth_series_is_refused(self):
        summary = self.summary()
        summary["by_context_depth"][0]["decode_tok_s"] = 999.0
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "by_context_depth"):
            self.ok()

    def test_a_non_finite_raw_row_is_refused(self):
        raw = (self.bundle / "workloads-spec-on.jsonl").read_text()
        old = json.loads(raw.splitlines()[0])["accept_rate"]
        (self.bundle / "workloads-spec-on.jsonl").write_text(
            raw.replace(f'"accept_rate": {old}', '"accept_rate": NaN', 1))
        with self.assertRaisesRegex(Invalid, "not a finite measurement"):
            self.ok()

    def test_an_errored_workload_row_is_refused(self):
        rows = rows_of("workloads-spec-on.jsonl")
        rows[0] = {"label": rows[0]["label"], "error": "connection refused"}
        self.write_rows("workloads-spec-on.jsonl", rows)
        with self.assertRaisesRegex(Invalid, "workload sweep"):
            self.ok()

    def test_the_workload_count_is_declared_not_derived(self):
        # Ten is the suite. A sweep that lost rows before assembly must fail
        # even though the file left behind is internally consistent.
        on = rows_of("workloads-spec-on.jsonl")[:8]
        labels = {r["label"] for r in on}
        off = [r for r in rows_of("workloads-spec-off.jsonl")
               if r["label"] in labels]
        self.write_rows("workloads-spec-on.jsonl", on)
        self.write_rows("workloads-spec-off.jsonl", off)
        summary = self.summary()
        summary["by_workload"] = summarise_workloads(on, off)
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "workload sweep"):
            self.ok()

    # ── the context sweep must be the whole sweep ──
    def test_a_single_arm_context_sweep_is_refused(self):
        rows = [r for r in rows_of_bundle(self.bundle, "context-sweep.jsonl")
                if r["config"] != "spec-off"]
        self.write_rows("context-sweep.jsonl", rows)
        with self.assertRaisesRegex(Invalid, "no spec-off rows"):
            self.ok()

    def test_every_required_depth_is_required_in_both_arms(self):
        for depth in pb.REQUIRED_DEPTHS:
            with self.subTest(depth=depth):
                rows = [r for r in
                        rows_of_bundle(self.bundle, "context-sweep.jsonl")
                        if not (r["target"] == depth and r["config"] == "spec-on")]
                self.write_rows("context-sweep.jsonl", rows)
                with self.assertRaisesRegex(Invalid, "missing depth"):
                    self.ok()
                self.setUp()

    def test_an_errored_or_non_positive_depth_row_is_refused(self):
        for mutate in ({"error": "timeout"}, {"decode_tps": 0.0},
                       {"prefill_tps": -1.0}):
            with self.subTest(mutate=mutate):
                rows = rows_of_bundle(self.bundle, "context-sweep.jsonl")
                rows[0].update(mutate)
                self.write_rows("context-sweep.jsonl", rows)
                with self.assertRaisesRegex(Invalid, "non-positive timing"):
                    self.ok()
                self.setUp()


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.bundle = make_bundle(self.dir / "b")
        self.addCleanup(self.tmp.cleanup)

    def test_the_archive_is_byte_identical_across_runs(self):
        # The no-clobber comparison is a digest comparison, so any timestamp in
        # the archive would make every re-run look like different content.
        first, first_sum = archive(self.bundle, self.dir / "a", "v2026.9.8")
        second, second_sum = archive(self.bundle, self.dir / "b2", "v2026.9.8")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(first_sum.read_text(), second_sum.read_text())

    def test_the_gzip_header_carries_no_timestamp_or_name(self):
        tarball, _ = archive(self.bundle, self.dir / "g", "v2026.9.8")
        header = tarball.read_bytes()[:10]
        self.assertEqual(int.from_bytes(header[4:8], "little"), 0)
        self.assertEqual(header[3] & 0x08, 0, "FNAME flag set in gzip header")

    def test_changed_content_changes_the_digest(self):
        # Guards against a determinism fix that makes every bundle identical.
        first, _ = archive(self.bundle, self.dir / "a", "v2026.9.8")
        (self.bundle / "host.json").write_text('{"changed": true}')
        second, _ = archive(self.bundle, self.dir / "c", "v2026.9.8")
        self.assertNotEqual(first.read_bytes(), second.read_bytes())

    def test_members_carry_no_identity(self):
        tarball, _ = archive(self.bundle, self.dir / "m", "v2026.9.8")
        with tarfile.open(tarball) as tar:
            for info in tar.getmembers():
                self.assertEqual((info.mtime, info.uid, info.gid, info.uname),
                                 (0, 0, 0, ""))

    def test_the_checksum_names_the_archive_and_matches_it(self):
        tarball, checksum = archive(self.bundle, self.dir / "s", "v2026.9.8")
        digest, name = checksum.read_text().split()
        self.assertEqual(name, tarball.name)
        self.assertEqual(digest, hashlib.sha256(tarball.read_bytes()).hexdigest())

    def test_a_symlink_in_the_bundle_is_refused(self):
        (self.bundle / "sneaky").symlink_to("/etc/passwd")
        with self.assertRaisesRegex(Invalid, "symlink"):
            archive(self.bundle, self.dir / "x", "v2026.9.8")

    def test_the_archive_may_not_be_written_inside_the_bundle(self):
        with self.assertRaisesRegex(Invalid, "contain itself"):
            archive(self.bundle, self.bundle / "out", "v2026.9.8")


class PublicationStateTest(unittest.TestCase):
    """gh is stubbed; these cover idempotence, partial repair and mismatch."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        bundle = make_bundle(self.dir / "b")
        self.tarball, self.checksum = archive(bundle, self.dir / "o", "v2026.9.8")
        self.work = self.dir / "w"
        self.work.mkdir()
        self.addCleanup(self.tmp.cleanup)

    def state(self, published: dict):
        """published maps asset name -> bytes already on the release."""
        def fake_download(cmd, **kw):
            name = cmd[cmd.index("--pattern") + 1]
            Path(cmd[cmd.index("--output") + 1]).write_bytes(published[name])
            return mock.Mock(returncode=0)

        with mock.patch.object(pb, "existing_assets",
                               return_value={n: {} for n in published}), \
             mock.patch.object(pb.subprocess, "run", side_effect=fake_download):
            return pb.publication_state("v2026.9.8", "o/r", self.tarball,
                                        self.checksum, self.work)

    def test_nothing_published_is_absent(self):
        self.assertEqual(self.state({}), "absent")

    def test_both_assets_matching_is_complete(self):
        self.assertEqual(self.state({
            self.tarball.name: self.tarball.read_bytes(),
            self.checksum.name: self.checksum.read_bytes()}), "complete")

    def test_an_interrupted_upload_reports_what_is_outstanding(self):
        # Checking only the archive reported this as complete and stranded the
        # release without its checksum, permanently.
        outstanding = self.state({self.tarball.name: self.tarball.read_bytes()})
        self.assertEqual([p.name for p in outstanding], [self.checksum.name])

    def test_different_published_content_is_refused(self):
        with self.assertRaisesRegex(Invalid, "DIFFERENT"):
            self.state({self.tarball.name: b"a different bundle"})

    def test_a_mismatched_checksum_alone_is_refused(self):
        self.assertRaisesRegex(
            Invalid, "DIFFERENT", self.state,
            {self.tarball.name: self.tarball.read_bytes(),
             self.checksum.name: b"0  wrong\n"})


if __name__ == "__main__":
    unittest.main()


class CommitBindingTest(unittest.TestCase):
    """--expected-commit must never substitute for the online resolution."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.bundle = make_bundle(self.dir / "b")
        self.addCleanup(self.tmp.cleanup)

    def run_cli(self, resolved, extra=()):
        argv = ["publish_bundle.py", "--bundle", str(self.bundle),
                "--release", "v2026.9.8", "--image-inspect", str(INSPECT),
                *extra]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(pb, "release_commit", return_value=resolved) \
                 as resolver, \
             mock.patch.object(pb, "publication_state",
                               return_value="complete"):
            return pb.main(), resolver

    def test_the_release_is_resolved_even_when_a_commit_is_supplied(self):
        rc, resolver = self.run_cli(COMMIT, ["--expected-commit", COMMIT])
        self.assertEqual(rc, 0)
        resolver.assert_called_once()

    def test_a_supplied_commit_that_disagrees_is_refused(self):
        rc, resolver = self.run_cli(COMMIT, ["--expected-commit", "f" * 40])
        self.assertEqual(rc, 1)
        resolver.assert_called_once()

    def test_an_image_from_another_commit_is_refused_online(self):
        rc, _ = self.run_cli("a" * 40)
        self.assertEqual(rc, 1)
