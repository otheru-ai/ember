#!/usr/bin/env bash
# Push the exact Forgejo event ref to Ember's public GitHub mirror.
#
# The credential is supplied through GIT_ASKPASS so it never appears in a
# remote URL, process argument, or git configuration. Only main and version
# tags are accepted: Forgejo remains the source of truth, while GitHub receives
# the refs that drive public CI and release publication.
set -euo pipefail

die() {
  printf 'mirror error: %s\n' "$*" >&2
  exit 1
}

source_sha="${GITHUB_SHA:-}"
source_ref="${GITHUB_REF:-}"
mirror_token="${EMBER_GITHUB_MIRROR_TOKEN:-}"
mirror_url="${EMBER_GITHUB_MIRROR_URL:-https://github.com/otheru-ai/ember.git}"

[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] ||
  die "GITHUB_SHA must be a full lowercase commit SHA"
case "$source_ref" in
  refs/heads/main | refs/tags/v[0-9]*) ;;
  *) die "ref is not eligible for the public mirror: $source_ref" ;;
esac
[[ -n "$mirror_token" ]] || die "EMBER_GITHUB_MIRROR_TOKEN is not set"

# A destination override exists solely for the offline regression test. Do not
# let workflow configuration redirect the write credential to another host.
if [[ -n "${EMBER_GITHUB_MIRROR_URL:-}" &&
      "${EMBER_MIRROR_TEST_MODE:-}" != "1" ]]; then
  die "EMBER_GITHUB_MIRROR_URL is allowed only in test mode"
fi

source_object="$(git rev-parse "$source_ref")" ||
  die "event ref is absent from the checkout: $source_ref"
source_commit="$(git rev-parse "$source_ref^{commit}")" ||
  die "event ref does not resolve to a commit: $source_ref"
[[ "$source_commit" = "$source_sha" ]] ||
  die "$source_ref resolves to $source_commit, not event SHA $source_sha"
[[ "$(git rev-parse 'HEAD^{commit}')" = "$source_sha" ]] ||
  die "checkout HEAD does not match event SHA $source_sha"

umask 077
askpass="$(mktemp)"
cleanup() { rm -f -- "$askpass"; }
trap cleanup EXIT
printf '%s\n' \
  '#!/bin/sh' \
  'case "$1" in' \
  '  *Username*) printf "%s\\n" x-access-token ;;' \
  '  *)          printf "%s\\n" "$EMBER_GITHUB_MIRROR_TOKEN" ;;' \
  'esac' >"$askpass"
chmod 700 "$askpass"

export GIT_ASKPASS="$askpass"
export GIT_TERMINAL_PROMPT=0
git push --porcelain "$mirror_url" "$source_ref:$source_ref"

remote_object="$(git ls-remote "$mirror_url" "$source_ref" | awk 'NR == 1 {print $1}')"
[[ "$remote_object" = "$source_object" ]] ||
  die "remote verification returned ${remote_object:-no ref}, expected $source_object"
printf 'mirrored %s at %s\n' "$source_ref" "$source_object"
