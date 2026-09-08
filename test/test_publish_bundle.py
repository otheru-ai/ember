#!/usr/bin/env python3
"""Contract for scripts/bench/publish_bundle.py. No GPU, no network, no gh."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

from publish_bundle import Invalid, REQUIRED, archive, validate  # noqa: E402


def make_bundle(directory: Path) -> Path:
    """A minimal bundle that must VALIDATE, so each test can break one thing."""
    bundle = directory / "ember-2026-09-08"
    bundle.mkdir()
    rows = [{"group": "decode-256", "ok": True, "decode_tokens_per_second": 41.5},
            {"group": "decode-256", "ok": True, "decode_tokens_per_second": 42.5},
            {"group": "prefill-4k", "prefill_tokens_per_second": 900.0,
             "evaluated_prefill_tokens": 4096}]
    (bundle / "raw-results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (bundle / "summary.json").write_text(json.dumps({
        "decode": {"samples": 2, "median_tps": 42.0},
        "prefill": {"prefill-4k": {"samples": 1, "median_tps": 900.0}},
    }))
    (bundle / "environment.json").write_text(json.dumps({
        "runtime": {"release": "2026.9.8", "container_image": "ghcr.io/x:v1"},
        "model": {"target_sha256": "a" * 64, "target_sha256_source": "computed",
                  "drafter_sha256": "b" * 64, "drafter_sha256_source": "computed"},
    }))
    for name in REQUIRED:
        if not (bundle / name).exists():
            (bundle / name).write_text("{}\n" if name.endswith(".json")
                                       else "row\n")
    return bundle


class PublishBundleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.bundle = make_bundle(self.dir)
        self.addCleanup(self.tmp.cleanup)

    def rewrite(self, name, obj):
        (self.bundle / name).write_text(json.dumps(obj))

    def test_a_complete_bundle_validates(self):
        env = validate(self.bundle, "v2026.9.8")
        self.assertEqual(env["runtime"]["release"], "2026.9.8")

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

    def test_missing_image_or_model_provenance_is_refused(self):
        for mutate, expect in (
            (lambda e: e["runtime"].pop("container_image"), "container_image"),
            (lambda e: e["model"].pop("target_sha256"), "target_sha256"),
            (lambda e: e["model"].update(drafter_sha256_source="guessed"),
             "computed or asserted"),
            (lambda e: e["model"].update(target_sha256_source="asserted"),
             "evidence reference"),
        ):
            with self.subTest(expect=expect):
                env = json.loads((self.bundle / "environment.json").read_text())
                mutate(env)
                self.rewrite("environment.json", env)
                with self.assertRaisesRegex(Invalid, expect):
                    validate(self.bundle, "v2026.9.8")
                self.setUp()

    def test_an_aggregate_without_matching_raw_rows_is_refused(self):
        # The check this replaced looked for a key summarise_groups never
        # writes, so it passed everything. Assert it can actually fail.
        summary = json.loads((self.bundle / "summary.json").read_text())
        summary["decode"]["samples"] = 99
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "does not match what is behind it"):
            validate(self.bundle, "v2026.9.8")

    def test_a_prefill_group_without_rows_is_refused(self):
        summary = json.loads((self.bundle / "summary.json").read_text())
        summary["prefill"]["prefill-64k"] = {"samples": 3}
        self.rewrite("summary.json", summary)
        with self.assertRaisesRegex(Invalid, "prefill-64k"):
            validate(self.bundle, "v2026.9.8")

    def test_a_non_finite_measurement_is_refused(self):
        raw = (self.bundle / "summary.json").read_text()
        (self.bundle / "summary.json").write_text(
            raw.replace('"median_tps": 42.0', '"median_tps": NaN'))
        with self.assertRaisesRegex(Invalid, "not a finite measurement"):
            validate(self.bundle, "v2026.9.8")

    def test_empty_raw_results_is_refused(self):
        (self.bundle / "raw-results.jsonl").write_text("")
        with self.assertRaisesRegex(Invalid, "raw-results.jsonl is empty"):
            validate(self.bundle, "v2026.9.8")

    def test_empty_workload_file_is_refused(self):
        (self.bundle / "workloads-spec-on.jsonl").write_text("\n  \n")
        with self.assertRaisesRegex(Invalid, "workloads-spec-on.jsonl is empty"):
            validate(self.bundle, "v2026.9.8")

    def test_the_archive_is_byte_identical_across_runs(self):
        # The no-clobber comparison is a digest comparison, so a timestamp
        # anywhere in the archive would make every re-run look like different
        # content and refuse to be idempotent.
        first, first_sum = archive(self.bundle, self.dir / "a", "v2026.9.8")
        second, second_sum = archive(self.bundle, self.dir / "b", "v2026.9.8")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(first_sum.read_text(), second_sum.read_text())

    def test_the_checksum_names_the_archive_and_matches_it(self):
        import hashlib
        tarball, checksum = archive(self.bundle, self.dir / "c", "v2026.9.8")
        digest, name = checksum.read_text().split()
        self.assertEqual(name, tarball.name)
        self.assertEqual(digest, hashlib.sha256(tarball.read_bytes()).hexdigest())

    def test_the_archive_holds_every_bundle_file_under_one_prefix(self):
        import tarfile
        tarball, _ = archive(self.bundle, self.dir / "d", "v2026.9.8")
        with tarfile.open(tarball) as tar:
            names = tar.getnames()
        for name in REQUIRED:
            self.assertIn(f"ember-2026.9.8-perf-bundle/{name}", names)


if __name__ == "__main__":
    unittest.main()
