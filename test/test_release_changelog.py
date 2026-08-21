#!/usr/bin/env python3
"""Tests for deterministic release metadata generation."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ci" / "release_changelog.py"


class ReleaseChangelogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self.temp.name)
        (self.repo / "ci").mkdir()
        shutil.copy2(SCRIPT, self.repo / "ci" / SCRIPT.name)
        (self.repo / "VERSION").write_text("2026.8.10\n")
        (self.repo / "CHANGELOG.md").write_text(
            "# Changelog\n\n## Unreleased\n\n- Curated operator note.\n\n"
            "## 2026.8.10\n\n- First release.\n"
        )
        (self.repo / "README.md").write_text(
            "Use ghcr.io/otheru-ai/ember:2026.8.10 here.\n"
        )
        (self.repo / "compose.yaml").write_text(
            "image: ghcr.io/otheru-ai/ember:2026.8.10\n"
        )
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "ci@example.invalid"], cwd=self.repo, check=True
        )
        subprocess.run(["git", "config", "user.name", "CI"], cwd=self.repo, check=True)
        self.commit("chore(release): v2026.8.10")
        subprocess.run(["git", "tag", "v2026.8.10"], cwd=self.repo, check=True)
        (self.repo / "feature").write_text("one\n")
        self.commit("feat(engine): add faster decode")
        (self.repo / "feature").write_text("two\n")
        self.commit("fix(server)!: reject malformed output")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def commit(self, subject: str) -> None:
        subprocess.run(["git", "add", "-A"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", subject], cwd=self.repo, check=True)

    def run_script(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["python3", str(self.repo / "ci" / SCRIPT.name), *args],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=check,
        )

    def test_prepare_moves_curated_notes_and_groups_commits(self) -> None:
        self.run_script("prepare", "--version", "2026.8.21")
        changelog = (self.repo / "CHANGELOG.md").read_text()
        self.assertIn("## Unreleased\n\n## 2026.8.21", changelog)
        self.assertIn("### Curated notes\n\n- Curated operator note.", changelog)
        self.assertIn("### Added\n\n- **engine:** add faster decode", changelog)
        self.assertIn("### Fixed\n\n- **Breaking:** **server:** reject malformed output", changelog)
        self.assertEqual((self.repo / "VERSION").read_text(), "2026.8.21\n")
        self.assertIn("ember:2026.8.21", (self.repo / "README.md").read_text())
        self.assertIn("ember:2026.8.21", (self.repo / "compose.yaml").read_text())
        changed = subprocess.check_output(
            ["git", "diff", "--name-only"], cwd=self.repo, text=True
        ).splitlines()
        self.assertEqual(
            sorted(changed), ["CHANGELOG.md", "README.md", "VERSION", "compose.yaml"]
        )

        notes = self.run_script("notes", "--version", "2026.8.21").stdout
        self.assertIn("Curated operator note", notes)
        self.assertNotIn("First release", notes)

    def test_rejects_invalid_or_non_increasing_version(self) -> None:
        invalid = self.run_script(
            "prepare", "--version", "2026.02.30", check=False
        )
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("expected YEAR.M.D", invalid.stderr)
        old = self.run_script("prepare", "--version", "2026.8.10", check=False)
        self.assertNotEqual(old.returncode, 0)
        self.assertIn("must be newer", old.stderr)


if __name__ == "__main__":
    unittest.main()
