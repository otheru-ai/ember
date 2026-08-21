#!/usr/bin/env python3
"""Mirror public GitHub issues into the Forgejo source-of-truth repository.

Forgejo is upstream here (`origin`) and GitHub is the published mirror, but
issues only ever arrive on the public side -- and issues are not git objects.
Nothing `git fetch` carries will ever bring them across; they are rows in a
forge database, reachable only through its API.

Forgejo cannot do this natively. Marking a repository as a pull mirror greys
out the issue/label/milestone migration options (forgejo#4962), and the request
to sync them on an interval (forgejo#5367) is open with no implementation --
"syncing issues & discussions will [be] some nice feature down the roadmap".
Upstream Gitea has carried the same request since 2017 (gitea#1876). Until one
of those lands, the copy has to be made by hand through both APIs.

Transport direction is forced by the network. git.otheru.ai resolves to a
tailnet address behind a Caddy-issued internal certificate, so GitHub cannot
deliver a webhook to it and a public ingress is not worth opening for this. The
self-hosted runner already attached to the public repository *is* on that
tailnet, so the sync runs there and reaches out: GitHub never needs a route in.
That makes the normal path event-driven rather than polled, with a scheduled
`--all` pass to reconcile whatever arrived while the runner was offline.

State lives in the mirrored text, not in a sidecar database. Every issue and
comment this script writes carries an HTML-comment marker naming its GitHub id,
so a re-run re-derives the whole mapping by reading Forgejo back. Re-running is
idempotent and losing the runner's disk costs nothing. Author-controlled text
is defanged with a zero-width space before embedding (see `defang`) so a
crafted issue body cannot claim to be the mirror of a different issue.

The event payload is trusted for exactly one field -- the issue number, which
is validated as an integer -- and every other value is refetched from the
GitHub API. Issue titles and bodies are attacker-controlled text on a public
repository, and this runs on a self-hosted runner; nothing from the payload
reaches a shell or a URL path unvalidated.

Nothing here writes to GitHub. The mirror is strictly one-way.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator

MARKER_NAME = "ember-gh-mirror"
MARKER_RE = re.compile(rf"<!-- {MARKER_NAME}:(issue|comment):(\d+) -->")
# Any marker-shaped text in author-controlled content, not just a well-formed
# one: the point is that nothing a stranger can type survives as a live marker.
SPOOF_RE = re.compile(re.escape(MARKER_NAME) + ":")

USER_AGENT = "ember-issue-mirror/1"
TIMEOUT = 30
MAX_ATTEMPTS = 4
RETRY_STATUS = {429, 500, 502, 503, 504}
PAGE_SIZE = 50
MAX_PAGES = 200  # runaway guard; 10k issues is far past anything plausible here

GITHUB_API = "https://api.github.com"


class MirrorError(RuntimeError):
    """A failure worth stopping the run for, reported as one actionable line."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@dataclass
class Forge:
    """Minimal JSON-over-HTTP client for one forge.

    stdlib only, deliberately: this runs on the builder runner, which has no
    project virtualenv and no reason to grow one for a sync job.
    """

    base: str
    auth: str
    page_param: str
    # Logged in place of `base`. Workflow logs on a public repository are
    # world-readable, and git.otheru.ai is not in public DNS -- printing the
    # full URL on the first 401 would publish an internal hostname. The path
    # alone carries every bit of diagnostic value the host does.
    name: str = "forge"
    accept: str = "application/json"
    context: ssl.SSLContext | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parts = urllib.parse.urlsplit(self.base)
        self._origin = f"{parts.scheme}://{parts.netloc}" if parts.netloc else ""

    def _scrub(self, text: str) -> str:
        """Strip the forge origin from anything bound for a log.

        Logging paths without a host is not sufficient on its own: a forge's
        error *body* echoes its own absolute URLs back at us -- Forgejo's 401
        carries `"url":"https://<host>/api/swagger"` -- and that text is
        reproduced verbatim in the exception. Public workflow logs must carry
        neither.
        """
        return text.replace(self._origin, f"<{self.name}>") if self._origin else text

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> Any:
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        payload = None if body is None else json.dumps(body).encode()

        for attempt in range(1, MAX_ATTEMPTS + 1):
            req = urllib.request.Request(url, data=payload, method=method)
            req.add_header("Authorization", self.auth)
            req.add_header("Accept", self.accept)
            req.add_header("User-Agent", USER_AGENT)
            if payload is not None:
                req.add_header("Content-Type", "application/json")
            for key, value in self.extra_headers.items():
                req.add_header(key, value)

            try:
                with urllib.request.urlopen(
                    req, timeout=TIMEOUT, context=self.context
                ) as resp:
                    raw = resp.read()
                return json.loads(raw) if raw else None
            except urllib.error.HTTPError as exc:
                if exc.code in RETRY_STATUS and attempt < MAX_ATTEMPTS:
                    time.sleep(_retry_delay(exc, attempt))
                    continue
                detail = self._scrub(
                    exc.read().decode("utf-8", "replace").strip()
                )[:400]
                raise MirrorError(
                    f"{self.name} {method} {path} -> HTTP {exc.code} "
                    f"{exc.reason}: {detail}"
                ) from exc
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, ssl.SSLCertVerificationError):
                    raise MirrorError(
                        f"{self.name} {method} {path} -> TLS verification "
                        f"failed: {self._scrub(str(exc.reason))}. "
                        "The Forgejo host uses an internal CA; export its root "
                        "certificate and point FORGEJO_CA_BUNDLE at the file "
                        "(see docs/ci.md)."
                    ) from exc
                if attempt < MAX_ATTEMPTS:
                    time.sleep(_retry_delay(None, attempt))
                    continue
                raise MirrorError(
                    f"{self.name} {method} {path} -> {self._scrub(str(exc.reason))}"
                ) from exc

        raise MirrorError(
            f"{self.name} {method} {path} -> exhausted {MAX_ATTEMPTS} attempts"
        )

    def paginate(
        self, path: str, *, params: dict[str, Any] | None = None
    ) -> Iterator[dict]:
        for page in range(1, MAX_PAGES + 1):
            query = dict(params or {})
            query["page"] = page
            query[self.page_param] = PAGE_SIZE
            batch = self.request("GET", path, params=query)
            if not isinstance(batch, list) or not batch:
                return
            yield from batch
            if len(batch) < PAGE_SIZE:
                return
        sys.stderr.write(
            f"warning: {path} still had results after {MAX_PAGES} pages; "
            "listing was truncated\n"
        )


def _retry_delay(exc: urllib.error.HTTPError | None, attempt: int) -> float:
    if exc is not None:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), 60.0)
    return min(2.0**attempt, 30.0)


def build_ssl_context() -> ssl.SSLContext | None:
    """Trust config for the Forgejo leg.

    Caddy issues git.otheru.ai a short-lived certificate from its own local
    authority, which is in no default trust store. Verification is therefore
    opt-in-configured rather than opt-out-disabled: FORGEJO_CA_BUNDLE is the
    supported path, and FORGEJO_INSECURE exists only so a first-run operator
    can prove the rest of the pipeline works before wiring up the CA.
    """
    if os.environ.get("FORGEJO_INSECURE") == "1":
        sys.stderr.write(
            "warning: FORGEJO_INSECURE=1 -- Forgejo TLS is unverified. "
            "Set FORGEJO_CA_BUNDLE instead for anything but a smoke test.\n"
        )
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    bundle = os.environ.get("FORGEJO_CA_BUNDLE", "").strip()
    if bundle:
        if not os.path.isfile(bundle):
            raise MirrorError(f"FORGEJO_CA_BUNDLE does not exist: {bundle}")
        return ssl.create_default_context(cafile=bundle)
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def defang(text: str) -> str:
    """Neutralise marker-shaped text so authored content cannot forge a mapping.

    A zero-width space inside the marker name breaks MARKER_RE while leaving
    the rendered comment invisible, so quoted markers still read correctly to a
    human looking at the raw markdown.
    """
    return SPOOF_RE.sub(MARKER_NAME + "​:", text or "")


def marker(kind: str, ident: int) -> str:
    return f"<!-- {MARKER_NAME}:{kind}:{ident} -->"


def read_marker(body: str | None, kind: str) -> int | None:
    """Last marker of `kind` in `body`, or None.

    Last rather than first: this script appends its marker at the end, so even
    against content written before defanging existed the trailing one is ours.
    """
    hits = [int(m.group(2)) for m in MARKER_RE.finditer(body or "") if m.group(1) == kind]
    return hits[-1] if hits else None


def _author(node: dict) -> tuple[str, str]:
    login = (node.get("user") or {}).get("login") or "ghost"
    return login, f"https://github.com/{urllib.parse.quote(login)}"


def render_issue_body(gh: dict, repo: str) -> str:
    login, profile = _author(gh)
    created = (gh.get("created_at") or "")[:10]
    header = (
        f"> Mirrored from GitHub **[{repo}#{gh['number']}]({gh['html_url']})**, "
        f"opened by [@{login}]({profile}) on {created}.\n"
        f"> Replies belong on GitHub -- comments written here are not sent back.\n"
    )
    body = defang(gh.get("body") or "").strip() or "_No description provided._"
    return f"{header}\n{body}\n\n{marker('issue', int(gh['number']))}"


def render_comment_body(gh: dict) -> str:
    login, profile = _author(gh)
    created = (gh.get("created_at") or "")[:10]
    header = (
        f"> [@{login}]({profile}) commented on GitHub on {created} "
        f"-- [view]({gh['html_url']})\n"
    )
    body = defang(gh.get("body") or "").strip() or "_(empty comment)_"
    return f"{header}\n{body}\n\n{marker('comment', int(gh['id']))}"


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


@dataclass
class Mirror:
    github: Forge
    forgejo: Forge
    gh_repo: str
    fj_repo: str
    dry_run: bool = False
    _labels: dict[str, int] | None = None
    _index: dict[int, dict] | None = None

    # -- Forgejo side lookups ------------------------------------------------

    @property
    def fj_path(self) -> str:
        return f"/repos/{self.fj_repo}"

    def index(self) -> dict[int, dict]:
        """GitHub issue number -> mirrored Forgejo issue, read back from markers.

        Built by listing rather than by the search `q` parameter: the volume is
        trivial, and listing does not depend on how the instance happens to have
        its issue indexer configured.
        """
        if self._index is None:
            found: dict[int, dict] = {}
            for issue in self.forgejo.paginate(
                f"{self.fj_path}/issues", params={"state": "all", "type": "issues"}
            ):
                number = read_marker(issue.get("body"), "issue")
                if number is not None:
                    found[number] = issue
            self._index = found
        return self._index

    def label_ids(self) -> dict[str, int]:
        if self._labels is None:
            self._labels = {
                label["name"].lower(): label["id"]
                for label in self.forgejo.paginate(f"{self.fj_path}/labels")
            }
        return self._labels

    def ensure_labels(self, gh_labels: list[dict]) -> list[str]:
        """Create any label this issue needs, and return the names to apply.

        Applying by name keeps one code path for create and update: Forgejo's
        EditIssueOption carries no `labels` field, but
        PUT /issues/{index}/labels accepts names as well as ids.
        """
        known = self.label_ids()
        names: list[str] = []
        for label in gh_labels:
            name = (label.get("name") or "").strip()
            if not name:
                continue
            names.append(name)
            if name.lower() in known:
                continue
            colour = (label.get("color") or "").strip().lstrip("#")
            body = {
                "name": name,
                "color": f"#{colour}" if re.fullmatch(r"[0-9a-fA-F]{6}", colour) else "#ededed",
                "description": (label.get("description") or "")[:255],
            }
            if self.dry_run:
                print(f"  would create label {name!r}")
                known[name.lower()] = -1
                continue
            created = self.forgejo.request("POST", f"{self.fj_path}/labels", body=body)
            known[name.lower()] = created["id"]
            print(f"  created label {name!r}")
        return names

    # -- one issue -----------------------------------------------------------

    def sync_issue(self, gh: dict) -> str:
        number = int(gh["number"])
        title = gh.get("title") or f"(untitled GitHub issue #{number})"
        wanted_body = render_issue_body(gh, self.gh_repo)
        wanted_state = "closed" if gh.get("state") == "closed" else "open"
        label_names = self.ensure_labels(gh.get("labels") or [])

        existing = self.index().get(number)
        if existing is None:
            action = self._create_issue(number, title, wanted_body, wanted_state)
            if self.dry_run:
                self._report_new_issue_details(number, label_names)
                return action
        else:
            action = self._update_issue(existing, title, wanted_body, wanted_state)

        target = self.index().get(number)
        if target is not None:
            children_changed = self._apply_labels(target, label_names)
            children_changed |= self._sync_comments(number, target)
            if children_changed and action == "unchanged":
                action = "would-update" if self.dry_run else "updated"
        return action

    def _report_new_issue_details(self, gh_number: int, names: list[str]) -> None:
        """Complete a new issue's dry-run without inventing a Forgejo index."""
        if names:
            print(f"  would set labels {sorted(names)}")
        for comment in self.github.paginate(
            f"/repos/{self.gh_repo}/issues/{gh_number}/comments"
        ):
            print(f"  would add comment {int(comment['id'])}")

    def _create_issue(self, number: int, title: str, body: str, state: str) -> str:
        if self.dry_run:
            print(f"issue #{number}: would create ({state}), with its labels and comments")
            return "would-create"
        created = self.forgejo.request(
            "POST",
            f"{self.fj_path}/issues",
            body={"title": title, "body": body, "closed": state == "closed"},
        )
        assert self._index is not None
        self._index[number] = created
        print(f"issue #{number}: created as {self.fj_repo}#{created['number']} ({state})")
        return "created"

    def _update_issue(self, existing: dict, title: str, body: str, state: str) -> str:
        changes: dict[str, Any] = {}
        if (existing.get("title") or "") != title:
            changes["title"] = title
        if (existing.get("body") or "") != body:
            changes["body"] = body
        if (existing.get("state") or "open") != state:
            changes["state"] = state
        if not changes:
            return "unchanged"

        index = existing["number"]
        if self.dry_run:
            print(f"{self.fj_repo}#{index}: would update {sorted(changes)}")
            return "would-update"
        updated = self.forgejo.request(
            "PATCH", f"{self.fj_path}/issues/{index}", body=changes
        )
        existing.update(updated or changes)
        print(f"{self.fj_repo}#{index}: updated {sorted(changes)}")
        return "updated"

    def _apply_labels(self, target: dict, names: list[str]) -> bool:
        current = sorted(
            (label.get("name") or "") for label in (target.get("labels") or [])
        )
        if current == sorted(names):
            return False
        index = target["number"]
        if self.dry_run:
            print(f"{self.fj_repo}#{index}: would set labels {sorted(names)}")
            return True
        self.forgejo.request(
            "PUT", f"{self.fj_path}/issues/{index}/labels", body={"labels": names}
        )
        target["labels"] = [{"name": n} for n in names]
        print(f"{self.fj_repo}#{index}: labels {sorted(names)}")
        return True

    def _sync_comments(self, gh_number: int, target: dict) -> bool:
        index = target["number"]
        changed = False
        mirrored: dict[int, dict] = {}
        for comment in self.forgejo.paginate(f"{self.fj_path}/issues/{index}/comments"):
            ident = read_marker(comment.get("body"), "comment")
            if ident is not None:
                mirrored[ident] = comment

        for comment in self.github.paginate(
            f"/repos/{self.gh_repo}/issues/{gh_number}/comments"
        ):
            ident = int(comment["id"])
            wanted = render_comment_body(comment)
            have = mirrored.get(ident)
            if have is None:
                changed = True
                if self.dry_run:
                    print(f"{self.fj_repo}#{index}: would add comment {ident}")
                    continue
                self.forgejo.request(
                    "POST",
                    f"{self.fj_path}/issues/{index}/comments",
                    body={"body": wanted},
                )
                print(f"{self.fj_repo}#{index}: added comment {ident}")
            elif (have.get("body") or "") != wanted:
                changed = True
                if self.dry_run:
                    print(f"{self.fj_repo}#{index}: would edit comment {ident}")
                    continue
                self.forgejo.request(
                    "PATCH",
                    f"{self.fj_path}/issues/comments/{have['id']}",
                    body={"body": wanted},
                )
                print(f"{self.fj_repo}#{index}: edited comment {ident}")
        return changed

    # -- selection -----------------------------------------------------------

    def fetch_issue(self, number: int) -> dict | None:
        gh = self.github.request("GET", f"/repos/{self.gh_repo}/issues/{number}")
        if gh is None or "pull_request" in gh:
            return None
        return gh

    def all_issues(self) -> Iterator[dict]:
        for gh in self.github.paginate(
            f"/repos/{self.gh_repo}/issues",
            params={"state": "all", "sort": "created", "direction": "asc"},
        ):
            if "pull_request" not in gh:
                yield gh


def issue_number_from_event(path: str) -> int:
    """The only value taken from the webhook payload, and it must be an integer.

    Everything else is refetched from the API, so no author-controlled string
    from the event ever reaches a URL, a shell, or a comparison here.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise MirrorError(f"cannot read event payload {path}: {exc}") from exc

    issue = payload.get("issue")
    if not isinstance(issue, dict):
        raise MirrorError(f"event payload {path} carries no issue object")
    if "pull_request" in issue:
        raise MirrorError("event is about a pull request, not an issue")
    number = issue.get("number")
    # bool is an int subclass in Python; reject JSON true/false explicitly.
    if type(number) is not int or number <= 0:
        raise MirrorError(f"event payload issue number is not a positive integer: {number!r}")
    return number


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MirrorError(f"{name} is not set")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--event-path", help="GitHub webhook payload; syncs that one issue")
    source.add_argument("--issue", type=int, help="Sync a single GitHub issue by number")
    source.add_argument("--all", action="store_true", help="Reconcile every issue")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report intended writes, change nothing"
    )
    args = parser.parse_args(argv)

    try:
        github = Forge(
            base=GITHUB_API,
            auth=f"Bearer {require_env('GITHUB_TOKEN')}",
            page_param="per_page",
            name="github",
            accept="application/vnd.github+json",
            extra_headers={"X-GitHub-Api-Version": "2022-11-28"},
        )
        forgejo = Forge(
            base=require_env("FORGEJO_URL").rstrip("/") + "/api/v1",
            auth=f"token {require_env('FORGEJO_TOKEN')}",
            page_param="limit",
            name="forgejo",
            context=build_ssl_context(),
        )
        mirror = Mirror(
            github=github,
            forgejo=forgejo,
            gh_repo=require_env("GITHUB_REPOSITORY"),
            fj_repo=require_env("FORGEJO_REPO"),
            dry_run=args.dry_run,
        )

        if args.all:
            targets = list(mirror.all_issues())
        else:
            number = (
                issue_number_from_event(args.event_path)
                if args.event_path
                else args.issue
            )
            gh = mirror.fetch_issue(number)
            if gh is None:
                print(f"issue #{number}: not an issue (pull request or deleted); nothing to do")
                return 0
            targets = [gh]

        tally: dict[str, int] = {}
        for gh in targets:
            action = mirror.sync_issue(gh)
            tally[action] = tally.get(action, 0) + 1
    except MirrorError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    summary = ", ".join(f"{count} {name}" for name, count in sorted(tally.items()))
    print(f"mirrored {len(targets)} issue(s) into {mirror.fj_repo}: {summary or 'nothing to do'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
