#!/usr/bin/env bash
#
# update-switcher.sh — merge main's and a release branch's copies of
# docs/_static/switcher.json into a single deduplicated, version-sorted
# switcher, and optionally mark one version as `preferred` (the version
# the docs "stable" switcher defaults to).
#
# Usage:
#   update-switcher.sh <NEW_VERSION> <RELEASE_BRANCH> <MARK_AS_LATEST>
#
# Arguments:
#   NEW_VERSION      Version being released, without a leading "v"
#                     (e.g. 1.4.2).
#   RELEASE_BRANCH    Name of the release branch to pull the other copy
#                      of switcher.json from (e.g. release/v1.4.2).
#   MARK_AS_LATEST     "true" to mark NEW_VERSION as `preferred` and clear
#                       that flag from every other entry; any other value
#                       leaves existing preferred flags untouched.
#
# Must be run from the root of the git repository (it writes to
# docs/_static/switcher.json relative to the current directory) with
# network access to fetch origin/main and origin/$RELEASE_BRANCH, and
# requires jq.
#
# Example:
#   ./update-switcher.sh 1.4.2 release/v1.4.2 true

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <NEW_VERSION> <RELEASE_BRANCH> <MARK_AS_LATEST>" >&2
  echo "  e.g. $(basename "$0") 1.4.2 release/v1.4.2 true" >&2
}

if [[ $# -ne 3 ]]; then
  usage
  exit 1
fi

NEW_VERSION="$1"
RELEASE_BRANCH="$2"
MARK_AS_LATEST="$3"

if [[ ! "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "NEW_VERSION must look like MAJOR.MINOR.PATCH (got: $NEW_VERSION)" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but was not found on PATH." >&2
  exit 1
fi

if [[ ! -d docs/_static ]]; then
  echo "docs/_static not found — run this from the repo root." >&2
  exit 1
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

MAIN_SWITCHER="$WORKDIR/main-switcher.json"
RELEASE_SWITCHER="$WORKDIR/release-switcher.json"
MERGED="$WORKDIR/switcher-merged.json"
PREFERRED="$WORKDIR/switcher-preferred.json"

# Rebuild switcher.json: union of both sides.
git show origin/main:docs/_static/switcher.json > "$MAIN_SWITCHER"
git show "origin/$RELEASE_BRANCH:docs/_static/switcher.json" > "$RELEASE_SWITCHER"
jq -s '
  (.[0] + .[1])
  | map(select(.version != "latest"))
  | unique_by(.version)
  | sort_by(.version | split(".") | map(tonumber? // 0))
  + [{"name":"latest","version":"latest","url":"https://deltares.github.io/hydromt/latest/"}]
' "$MAIN_SWITCHER" "$RELEASE_SWITCHER" > "$MERGED"

# Mark this version preferred (the "stable" docs default) if requested.
# unique_by above already keeps main's copy of any entry present on both
# sides, so existing preferred flags for other versions survive untouched
# whenever MARK_AS_LATEST is false.
if [[ "$MARK_AS_LATEST" == "true" ]]; then
  jq --arg v "$NEW_VERSION" '
    map(
      if .version == "latest" then .
      else .preferred = (.version == $v)
      end
    )
  ' "$MERGED" > "$PREFERRED"

  PREFERRED_COUNT=$(jq '[.[] | select(.version != "latest" and .preferred == true)] | length' "$PREFERRED")
  if [[ "$PREFERRED_COUNT" -ne 1 ]]; then
    echo "ERROR: switcher.json has $PREFERRED_COUNT preferred entries, expected exactly 1." >&2
    exit 1
  fi

  mv "$PREFERRED" "$MERGED"
fi

cp "$MERGED" docs/_static/switcher.json

echo "docs/_static/switcher.json updated with v$NEW_VERSION." >&2
