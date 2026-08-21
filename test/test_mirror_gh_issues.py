#!/usr/bin/env python3
"""Offline regression tests for the GitHub-to-Forgejo issue mirror."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mirror_gh_issues", ROOT / "ci" / "mirror_gh_issues.py"
)
assert SPEC and SPEC.loader
mirror_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mirror_module
SPEC.loader.exec_module(mirror_module)


class FakeForge:
    def __init__(self, pages: dict[str, list[dict]] | None = None) -> None:
        self.pages = pages or {}
        self.requests: list[tuple[str, str, dict | None]] = []
        self.responses: dict[tuple[str, str], dict] = {}

    def paginate(self, path: str, *, params: dict | None = None):
        del params
        yield from self.pages.get(path, [])

    def request(
        self, method: str, path: str, *, params: dict | None = None, body=None
    ):
        del params
        self.requests.append((method, path, body))
        return self.responses.get((method, path))


def github_issue(number: int = 7) -> dict:
    return {
        "number": number,
        "title": "Public bug",
        "body": "A body with ember-gh-mirror:issue:999 in it",
        "state": "open",
        "created_at": "2026-08-20T12:00:00Z",
        "html_url": f"https://github.com/otheru-ai/ember/issues/{number}",
        "user": {"login": "reporter"},
        "labels": [{"name": "bug", "color": "ff0000", "description": "Bug"}],
    }


def github_comment(ident: int = 101, body: str = "More detail") -> dict:
    return {
        "id": ident,
        "body": body,
        "created_at": "2026-08-20T13:00:00Z",
        "html_url": "https://github.com/otheru-ai/ember/issues/7#issuecomment-101",
        "user": {"login": "commenter"},
    }


class MirrorTests(unittest.TestCase):
    def test_author_text_cannot_spoof_mapping_marker(self) -> None:
        rendered = mirror_module.render_issue_body(github_issue(), "otheru-ai/ember")
        self.assertEqual(mirror_module.read_marker(rendered, "issue"), 7)
        self.assertNotIn("ember-gh-mirror:issue:999", rendered)

    def test_create_copies_label_and_comment(self) -> None:
        gh = FakeForge(
            {"/repos/otheru-ai/ember/issues/7/comments": [github_comment()]}
        )
        fj = FakeForge(
            {
                "/repos/otheru/ember/issues": [],
                "/repos/otheru/ember/labels": [],
                "/repos/otheru/ember/issues/42/comments": [],
            }
        )
        fj.responses[("POST", "/repos/otheru/ember/labels")] = {
            "id": 3,
            "name": "bug",
        }
        fj.responses[("POST", "/repos/otheru/ember/issues")] = {
            "number": 42,
            "title": "Public bug",
            "body": mirror_module.render_issue_body(github_issue(), "otheru-ai/ember"),
            "state": "open",
            "labels": [],
        }
        mirror = mirror_module.Mirror(gh, fj, "otheru-ai/ember", "otheru/ember")

        with contextlib.redirect_stdout(io.StringIO()):
            action = mirror.sync_issue(github_issue())

        self.assertEqual(action, "created")
        self.assertIn(
            (
                "PUT",
                "/repos/otheru/ember/issues/42/labels",
                {"labels": ["bug"]},
            ),
            fj.requests,
        )
        self.assertTrue(
            any(
                method == "POST"
                and path == "/repos/otheru/ember/issues/42/comments"
                and mirror_module.read_marker(body["body"], "comment") == 101
                for method, path, body in fj.requests
            )
        )

    def test_comment_only_change_counts_as_update(self) -> None:
        issue = github_issue()
        fj_issue = {
            "number": 42,
            "title": issue["title"],
            "body": mirror_module.render_issue_body(issue, "otheru-ai/ember"),
            "state": "open",
            "labels": [{"name": "bug"}],
        }
        gh = FakeForge(
            {"/repos/otheru-ai/ember/issues/7/comments": [github_comment()]}
        )
        fj = FakeForge(
            {
                "/repos/otheru/ember/issues": [fj_issue],
                "/repos/otheru/ember/labels": [{"id": 3, "name": "bug"}],
                "/repos/otheru/ember/issues/42/comments": [],
            }
        )
        mirror = mirror_module.Mirror(gh, fj, "otheru-ai/ember", "otheru/ember")

        with contextlib.redirect_stdout(io.StringIO()):
            action = mirror.sync_issue(issue)

        self.assertEqual(action, "updated")

    def test_new_issue_dry_run_reports_children_without_writes(self) -> None:
        gh = FakeForge(
            {"/repos/otheru-ai/ember/issues/7/comments": [github_comment()]}
        )
        fj = FakeForge(
            {
                "/repos/otheru/ember/issues": [],
                "/repos/otheru/ember/labels": [],
            }
        )
        mirror = mirror_module.Mirror(
            gh, fj, "otheru-ai/ember", "otheru/ember", dry_run=True
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            action = mirror.sync_issue(github_issue())

        self.assertEqual(action, "would-create")
        self.assertIn("would set labels ['bug']", output.getvalue())
        self.assertIn("would add comment 101", output.getvalue())
        self.assertEqual(fj.requests, [])

    def test_event_accepts_only_positive_integer_issue_number(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event = pathlib.Path(temp_dir) / "event.json"
            event.write_text(json.dumps({"issue": {"number": 7}}), encoding="utf-8")
            self.assertEqual(mirror_module.issue_number_from_event(str(event)), 7)

            event.write_text(json.dumps({"issue": {"number": "7"}}), encoding="utf-8")
            with self.assertRaises(mirror_module.MirrorError):
                mirror_module.issue_number_from_event(str(event))

            event.write_text(json.dumps({"issue": {"number": True}}), encoding="utf-8")
            with self.assertRaises(mirror_module.MirrorError):
                mirror_module.issue_number_from_event(str(event))


if __name__ == "__main__":
    unittest.main()
