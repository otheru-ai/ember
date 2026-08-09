#!/usr/bin/env python3
"""Contract tests for the history-free development promotion tool."""

from __future__ import annotations

import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "promote_from_dev.py"


def run(*args: str, cwd: pathlib.Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args), cwd=cwd, text=True, capture_output=True, check=check
    )


def init_repo(path: pathlib.Path) -> None:
    path.mkdir()
    run("git", "init", "-q", cwd=path)
    run("git", "config", "user.name", "Promotion Test", cwd=path)
    run("git", "config", "user.email", "promotion@example.invalid", cwd=path)


def commit_all(path: pathlib.Path, message: str) -> str:
    run("git", "add", "-A", cwd=path)
    run("git", "commit", "-q", "-m", message, cwd=path)
    return run("git", "rev-parse", "HEAD", cwd=path).stdout.strip()


class PromotionTests(unittest.TestCase):
    def test_three_way_merge_policy_and_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            dev, release = root / "ember-dev", root / "ember"
            init_repo(dev)
            (dev / "src").mkdir()
            (dev / "docs").mkdir()
            (dev / "src" / "unit.c").write_text("release-line\nshared\ndev-line\n")
            (dev / "src" / "conflict.c").write_text("one\ntwo\nthree\nfour\nfive\n")
            (dev / "docs" / "note.md").write_text("old\n")
            base = commit_all(dev, "base")

            init_repo(release)
            (release / "src").mkdir()
            (release / ".release").mkdir()
            (release / "src" / "unit.c").write_text(
                "release-line-local\nshared\ndev-line\n"
            )
            (release / "src" / "conflict.c").write_text(
                "release-one\ntwo\nthree\nfour\nrelease-five\n"
            )
            (release / ".release" / "ember-dev-base").write_text(base + "\n")
            (release / ".release" / "promotion-log.md").write_text("# Log\n")
            commit_all(release, "release base")

            (dev / "src" / "unit.c").write_text(
                "release-line\nshared\ndev-line-new\n"
            )
            (dev / "src" / "conflict.c").write_text(
                "dev-one\ntwo\nthree\nfour\ndev-five\n"
            )
            (dev / "docs" / "note.md").write_text("internal note\n")
            (dev / "reports").mkdir()
            (dev / "reports" / "private.txt").write_text("private\n")
            target = commit_all(dev, "development update")

            plan = run(
                str(SCRIPT), "--release", str(release), "--dev", str(dev),
                cwd=release,
            )
            self.assertIn("automatic include", plan.stdout)
            self.assertIn("manual    unresolved", plan.stdout)
            self.assertIn("blocked   unresolved", plan.stdout)
            self.assertIn("clean-merge", plan.stdout)
            self.assertIn("conflict", plan.stdout)
            self.assertIn("conflicts require a manual port", plan.stdout)

            refused = run(
                str(SCRIPT), "--release", str(release), "--dev", str(dev),
                "--apply", cwd=release, check=False,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("unresolved manual paths", refused.stderr)

            applied = run(
                str(SCRIPT), "--release", str(release), "--dev", str(dev),
                "--skip", "docs/note.md", "--skip", "reports/private.txt",
                "--skip", "src/conflict.c",
                "--apply", cwd=release,
            )
            self.assertIn("applied:", applied.stdout)
            self.assertEqual(
                (release / "src" / "unit.c").read_text(),
                "release-line-local\nshared\ndev-line-new\n",
            )
            self.assertFalse((release / "reports" / "private.txt").exists())
            self.assertEqual(
                (release / ".release" / "ember-dev-base").read_text().strip(),
                target,
            )
            ledger = (release / ".release" / "promotion-log.md").read_text()
            self.assertIn("docs/note.md", ledger)
            self.assertIn("reports/private.txt", ledger)


if __name__ == "__main__":
    unittest.main()
