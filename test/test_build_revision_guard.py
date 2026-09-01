#!/usr/bin/env python3
"""Regression tests for build-time release-tool revision binding."""

import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


if len(sys.argv) > 2:
    CMAKE = pathlib.Path(sys.argv.pop(1)).resolve()
    GUARD = pathlib.Path(sys.argv.pop(1)).resolve()
else:
    CMAKE = None
    GUARD = None


class BuildRevisionGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        if CMAKE is None or GUARD is None:
            self.fail("usage: test_build_revision_guard.py <cmake> <guard-script>")
        self.git = shutil.which("git")
        if self.git is None:
            self.skipTest("git unavailable")
        self.temporary = tempfile.TemporaryDirectory(
            prefix="ember-build-revision-guard-")
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        subprocess.run([self.git, "init", "-q", str(self.root)], check=True)
        subprocess.run(
            [self.git, "-C", str(self.root), "config", "user.email",
             "revision-guard@example.invalid"], check=True)
        subprocess.run(
            [self.git, "-C", str(self.root), "config", "user.name",
             "Revision Guard"], check=True)
        (self.root / "tracked").write_text("bound\n", encoding="utf-8")
        subprocess.run(
            [self.git, "-C", str(self.root), "add", "tracked"], check=True)
        subprocess.run(
            [self.git, "-C", str(self.root), "commit", "-q", "-m", "fixture"],
            check=True)
        self.revision = subprocess.run(
            [self.git, "-C", str(self.root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()

    def run_guard(self, source: pathlib.Path, revision: str,
                  git: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(CMAKE), f"-DEMBER_EXPECTED_GIT_HEAD={revision}",
             f"-DEMBER_SOURCE_DIR={source}",
             f"-DEMBER_GIT_EXECUTABLE={git if git is not None else self.git}",
             "-P", str(GUARD)],
            capture_output=True, text=True, check=False)

    def test_matching_checkout_passes(self) -> None:
        result = self.run_guard(self.root, self.revision)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stale_checkout_fails(self) -> None:
        result = self.run_guard(self.root, "0" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stale Ember build directory", result.stderr)

    def test_exported_source_without_git_passes(self) -> None:
        exported = self.root / "exported"
        exported.mkdir()
        result = self.run_guard(exported, self.revision, git="")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
