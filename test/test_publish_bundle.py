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

IMAGE = "ghcr.io/otheru-ai/ember@sha256:" + "c" * 64


def make_bundle(directory: Path) -> Path:
    """A bundle that VALIDATES, so each test can break exactly one thing.

    The summary is produced by the assembler's own summarisers rather than
    hand-written, which is the point: publish_bundle recomputes it the same way
    and compares, so a hand-written summary would only prove the fixture and
    the tool agreed on a constant.
    """
    bundle = directory / "ember-2026-09-08"
    bundle.mkdir(parents=True)
    rows = [{"group": "decode-256", "ok": True, "decode_tokens_per_second": 41.5,
             "accept_rate": 0.7, "completion_tokens": 256},
            {"group": "decode-256", "ok": True, "decode_tokens_per_second": 42.5,
             "accept_rate": 0.8, "completion_tokens": 256},
            {"group": "prefill-4k", "prefill_tokens_per_second": 900.0,
             "evaluated_prefill_tokens": 4096},
            {"kind": "summary", "vision": {"images": 4, "ok": True}}]
    wl_on = [{"label": "chat", "decode_tps": 40.0, "prefill_tps": 800.0,
              "accept_rate": 0.7, "spec_ran": True, "spec_cycles": 12},
             {"label": "code", "decode_tps": 38.0, "prefill_tps": 780.0,
              "accept_rate": 0.6, "spec_ran": True, "spec_cycles": 9}]
    wl_off = [{"label": "chat", "decode_tps": 20.0, "prefill_tps": 790.0,
               "accept_rate": 0.0, "spec_ran": False, "spec_cycles": 0},
              {"label": "code", "decode_tps": 19.0, "prefill_tps": 770.0,
               "accept_rate": 0.0, "spec_ran": False, "spec_cycles": 0}]
    ctx = [{"config": c, "target": t, "prompt_tokens": t,
            "prefill_tps": 900.0, "decode_tps": d, "accept": 0.7}
           for t in (4096, 16384) for c, d in (("spec-on", 40.0),
                                               ("spec-off", 20.0))]

    def write(name, records):
        (bundle / name).write_text(
            "".join(json.dumps(r) + "\n" for r in records))

    write("raw-results.jsonl", rows)
    write("workloads-spec-on.jsonl", wl_on)
    write("workloads-spec-off.jsonl", wl_off)
    write("context-sweep.jsonl", ctx)

    summary = summarise_groups(rows)
    summary["by_workload"] = summarise_workloads(wl_on, wl_off)
    summary["by_context_depth"] = summarise_context(ctx)
    summary["vision"] = rows[-1]["vision"]
    (bundle / "summary.json").write_text(json.dumps(summary))
    (bundle / "environment.json").write_text(json.dumps({
        "runtime": {"release": "2026.9.8", "container_image": IMAGE},
        "model": {"target_sha256": "a" * 64, "target_sha256_source": "computed",
                  "drafter_sha256": "b" * 64, "drafter_sha256_source": "computed"},
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

    def rewrite(self, name, obj):
        (self.bundle / name).write_text(json.dumps(obj))

    def summary(self):
        return json.loads((self.bundle / "summary.json").read_text())

    def environment(self):
        return json.loads((self.bundle / "environment.json").read_text())

    def test_a_complete_bundle_validates(self):
        env = validate(self.bundle, "v2026.9.8")
        self.assertEqual(env["_publication"]["image_digest"], "c" * 64)

    def test_tag_may_carry_the_v_prefix_or_not(self):
        validate(self.bundle, "2026.9.8")

    def test_a_bundle_measured_on_another_release_is_refused(self):
        with self.assertRaisesRegex(Invalid, "did not measure"):
            validate(self.bundle, "v2026.9.5")

    def test_every_required_file_is_required(self):
        for name in REQUIRED:
            with self.subTest(name=name):
                shutil.copy(self.bundle / name, self.dir / "held")
                (self.bundle / name).unlink()
                with self.assertRaisesRegex(Invalid, "incomplete bundle"):
                    validate(self.bundle, "v2026.9.8")
                shutil.copy(self.dir / "held", self.bundle / name)

    def test_a_mutable_tag_without_a_digest_is_refused(self):
        env = self.environment()
        env["runtime"]["container_image"] = "ghcr.io/otheru-ai/ember:v2026.9.8"
        self.rewrite("environment.json", env)
        with self.assertRaisesRegex(Invalid, "mutable tag with no digest"):
            validate(self.bundle, "v2026.9.8")
        # ...and accepted when the operator supplies the captured digest.
        env2 = validate(self.bundle, "v2026.9.8", image_digest="sha256:" + "d" * 64)
        self.assertEqual(env2["_publication"]["image_digest"], "d" * 64)

    def test_a_supplied_digest_may_not_contradict_the_bundle(self):
        with self.assertRaisesRegex(Invalid, "contradicts"):
            validate(self.bundle, "v2026.9.8", image_digest="sha256:" + "e" * 64)

    def test_model_digests_must_be_real_sha256(self):
        for value, expect in (("not-a-digest", "full lowercase hex"),
                              ("A" * 64, "full lowercase hex"),
                              ("a" * 63, "full lowercase hex")):
            with self.subTest(value=value):
                env = self.environment()
                env["model"]["target_sha256"] = value
                self.rewrite("environment.json", env)
                with self.assertRaisesRegex(Invalid, expect):
                    validate(self.bundle, "v2026.9.8")

    def test_asserted_digest_needs_evidence(self):
        env = self.environment()
        env["model"]["drafter_sha256_source"] = "asserted"
        self.rewrite("environment.json", env)
        with self.assertRaisesRegex(Invalid, "evidence reference"):
            validate(self.bundle, "v2026.9.8")

    def test_an_invented_aggregate_with_matching_counts_is_refused(self):
        # The count check alone passed this: same samples, fabricated median.
        summary = self.summary()
        summary["decode"]["median_tps"] = 99.0
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "not derived from these rows"):
            validate(self.bundle, "v2026.9.8")

    def test_an_invented_workload_table_is_refused(self):
        summary = self.summary()
        summary["by_workload"]["chat"]["speedup"] = 9.9
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "by_workload"):
            validate(self.bundle, "v2026.9.8")

    def test_an_invented_depth_series_is_refused(self):
        summary = self.summary()
        summary["by_context_depth"][0]["decode_tok_s"] = 999.0
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "by_context_depth"):
            validate(self.bundle, "v2026.9.8")

    def test_a_missing_prefill_group_is_refused(self):
        rows = [json.loads(l) for l in
                (self.bundle / "raw-results.jsonl").read_text().splitlines()]
        keep = [r for r in rows if r.get("group") != "prefill-4k"]
        (self.bundle / "raw-results.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in keep))
        with self.assertRaisesRegex(Invalid, "no usable prefill"):
            validate(self.bundle, "v2026.9.8")

    def test_a_missing_vision_result_is_refused(self):
        summary = self.summary()
        del summary["vision"]
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "no vision results"):
            validate(self.bundle, "v2026.9.8")

    def test_a_non_finite_raw_row_is_refused(self):
        # A NaN that the median averages away still means a broken run, so the
        # raw rows are checked and not only the aggregate.
        raw = (self.bundle / "workloads-spec-on.jsonl").read_text()
        (self.bundle / "workloads-spec-on.jsonl").write_text(
            raw.replace('"accept_rate": 0.7', '"accept_rate": NaN'))
        with self.assertRaisesRegex(Invalid, "not a finite measurement"):
            validate(self.bundle, "v2026.9.8")

    def test_an_errored_workload_row_is_refused(self):
        rows = [json.loads(l) for l in
                (self.bundle / "workloads-spec-on.jsonl").read_text().splitlines()]
        rows[0] = {"label": "chat", "error": "connection refused"}
        (self.bundle / "workloads-spec-on.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(Invalid, "workload sweep"):
            validate(self.bundle, "v2026.9.8")

    def test_a_short_workload_sweep_is_refused_when_the_count_is_declared(self):
        with self.assertRaisesRegex(Invalid, "workload sweep"):
            validate(self.bundle, "v2026.9.8", expected_workloads=5)

    def test_a_single_arm_context_sweep_is_refused(self):
        rows = [json.loads(l) for l in
                (self.bundle / "context-sweep.jsonl").read_text().splitlines()]
        keep = [r for r in rows if r["config"] != "spec-off"]
        (self.bundle / "context-sweep.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in keep))
        with self.assertRaisesRegex(Invalid, "no spec-off rows"):
            validate(self.bundle, "v2026.9.8")

    def test_a_one_point_context_sweep_is_refused(self):
        rows = [json.loads(l) for l in
                (self.bundle / "context-sweep.jsonl").read_text().splitlines()]
        keep = [r for r in rows if r["target"] == 4096]
        (self.bundle / "context-sweep.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in keep))
        with self.assertRaisesRegex(Invalid, "at least two"):
            validate(self.bundle, "v2026.9.8")


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
