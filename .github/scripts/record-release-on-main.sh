#!/usr/bin/env bash
# Record a release tagged on an older release family's branch (a
# `release/vX.Y` branch whose MAJOR.MINOR is older than `main`'s) by
# forwarding its changelog and switcher.json entries to `main`. The version
# in source (hydromt/__init__.py) is intentionally NOT touched, because
# main is preparing a newer release family and must keep its own version.
#
# A "release family" is the set of releases sharing the same MAJOR.MINOR.
# Example: the 1.4 family contains v1.4.0, v1.4.1, v1.4.2 ... and lives on
# `release/v1.4`.
#
# Usage: record-release-on-main.sh <RELEASE_BRANCH> <NEW_VERSION>
#
# Prerequisites:
#  - origin/main and the release branch are fetched.
#  - git is configured with a committer identity.
#  - jq and awk are available.
#
# Effect:
#  - Checks out a new branch `record-release/v<NEW_VERSION>` from origin/main.
#  - Inserts the v<NEW_VERSION> section into docs/changelog.rst above the
#    matching X.Y.0 entry (or above the first version heading as fallback).
#  - Merges the switcher.json entry for v<NEW_VERSION> into
#    docs/_static/switcher.json.
#  - Commits and pushes the branch.
#  - Echoes the branch name to stdout.

set -euo pipefail

RELEASE_BRANCH="${1:-}"
NEW_VERSION="${2:-}"

if [[ -z "$RELEASE_BRANCH" || -z "$NEW_VERSION" ]]; then
  echo "Usage: record-release-on-main.sh <RELEASE_BRANCH> <NEW_VERSION>" >&2
  exit 1
fi

RECORD_BRANCH="record-release/v$NEW_VERSION"

git checkout -B "$RECORD_BRANCH" "origin/main"

# Extract the v<NEW_VERSION> section from the release branch's changelog.
git show "origin/$RELEASE_BRANCH:docs/changelog.rst" > /tmp/release-changelog.rst
awk -v ver="v$NEW_VERSION" '
  $0 ~ "^"ver"( |$)" { capture = 1 }
  capture && /^v[0-9]+\.[0-9]+\.[0-9]+/ && $0 !~ "^"ver"( |$)" { exit }
  capture { print }
' /tmp/release-changelog.rst > /tmp/section.rst

if [ ! -s /tmp/section.rst ]; then
  echo "Could not extract v$NEW_VERSION section from release branch changelog." >&2
  exit 1
fi

# Insert the section into main's changelog above the matching X.Y.0 heading
# (the heading that opens the same release family).
MAJOR=$(echo "$NEW_VERSION" | cut -d. -f1)
MINOR=$(echo "$NEW_VERSION" | cut -d. -f2)
FAMILY_BASE="v${MAJOR}.${MINOR}.0"

awk -v family_base="$FAMILY_BASE" -v section_file="/tmp/section.rst" '
  BEGIN {
    while ((getline line < section_file) > 0) section = section line "\n"
    close(section_file)
    inserted = 0
  }
  !inserted && $0 ~ "^"family_base"( |$)" {
    printf "%s\n", section
    inserted = 1
  }
  { print }
  END {
    if (!inserted) {
      # Fallback: append at the end of the file.
      printf "\n%s", section
    }
  }
' docs/changelog.rst > /tmp/changelog.rst
mv /tmp/changelog.rst docs/changelog.rst

# Merge switcher.json: union, dedup by version, sort numerically, keep 'latest' last.
git show "origin/$RELEASE_BRANCH:docs/_static/switcher.json" > /tmp/release-switcher.json
jq -s '
  (.[0] + .[1])
  | map(select(.version != "latest"))
  | unique_by(.version)
  | sort_by(.version | split(".") | map(tonumber? // 0))
  + [{"name":"latest","version":"latest","url":"https://deltares.github.io/hydromt/latest/"}]
' docs/_static/switcher.json /tmp/release-switcher.json > /tmp/switcher.json
mv /tmp/switcher.json docs/_static/switcher.json

git add docs/changelog.rst docs/_static/switcher.json
git commit -m "Record v$NEW_VERSION release on main (changelog and switcher)"
git push --set-upstream origin "$RECORD_BRANCH"

echo "$RECORD_BRANCH"
