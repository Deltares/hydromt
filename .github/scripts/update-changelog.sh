#!/usr/bin/env bash
#
# update-changelog.sh — merge a release branch's changelog section into
# main's copy of docs/changelog.rst, in descending version order, and
# optionally reset the Unreleased section on main's copy first.
#
# Usage:
#   update-changelog.sh <NEW_VERSION> <RELEASE_BRANCH> <VERSION_BUMPED>
#
# Arguments:
#   NEW_VERSION      Version being released, without a leading "v"
#                     (e.g. 1.4.2).
#   RELEASE_BRANCH    Name of the release branch to pull the new
#                      changelog section from (e.g. release/v1.4.2).
#   VERSION_BUMPED    "true" to reset main's Unreleased section back to
#                      an empty template before merging; any other value
#                      leaves it as-is.
#
# Must be run from the root of the git repository (it writes to
# docs/changelog.rst relative to the current directory) with network
# access to fetch origin/main and origin/$RELEASE_BRANCH.
#
# Example:
#   ./update-changelog.sh 1.4.2 release/v1.4.2 true

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <NEW_VERSION> <RELEASE_BRANCH> <VERSION_BUMPED>" >&2
  echo "  e.g. $(basename "$0") 1.4.2 release/v1.4.2 true" >&2
}

if [[ $# -ne 3 ]]; then
  usage
  exit 1
fi

NEW_VERSION="$1"
RELEASE_BRANCH="$2"
VERSION_BUMPED="$3"

if [[ ! "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "NEW_VERSION must look like MAJOR.MINOR.PATCH (got: $NEW_VERSION)" >&2
  exit 1
fi

MAJOR="${NEW_VERSION%%.*}"
REST="${NEW_VERSION#*.}"
MINOR="${REST%%.*}"
NEW_PATCH="${NEW_VERSION##*.}"

if [[ ! -f docs/changelog.rst ]]; then
  echo "docs/changelog.rst not found — run this from the repo root." >&2
  exit 1
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

MAIN_CHANGELOG="$WORKDIR/main-changelog.rst"
RELEASE_CHANGELOG="$WORKDIR/release-changelog.rst"
SECTION="$WORKDIR/section.rst"
MERGED="$WORKDIR/changelog-merged.rst"

# Rebuild changelog from main's copy.
git show origin/main:docs/changelog.rst > "$MAIN_CHANGELOG"

# If version was bumped, replace the Unreleased section with a fresh one.
if [[ "$VERSION_BUMPED" == "true" ]]; then
  awk '
    BEGIN { skip = 0; printed_header = 0 }
    /^Unreleased$/ { skip = 1; next }
    skip && /^=+$/ { next }
    skip && /^v[0-9]+\.[0-9]+\.[0-9]+/ { skip = 0 }
    skip { next }
    !printed_header {
      print "Unreleased"
      print "=========="
      print ""
      print "New"
      print "---"
      print ""
      print "Changed"
      print "-------"
      print ""
      print "Fixed"
      print "-----"
      print ""
      print "Deprecated"
      print "----------"
      print ""
      print "Removed"
      print "-------"
      print ""
      printed_header = 1
    }
    { print }
  ' "$MAIN_CHANGELOG" > "$WORKDIR/main-changelog-fresh.rst"
  mv "$WORKDIR/main-changelog-fresh.rst" "$MAIN_CHANGELOG"
fi

# Extract the v<NEW_VERSION> section from the release branch changelog.
git show "origin/$RELEASE_BRANCH:docs/changelog.rst" > "$RELEASE_CHANGELOG"
awk -v ver="v$NEW_VERSION" '
  $0 ~ "^"ver"( |$)" { capture = 1 }
  capture && /^v[0-9]+\.[0-9]+\.[0-9]+/ && $0 !~ "^"ver"( |$)" { exit }
  capture { print }
' "$RELEASE_CHANGELOG" > "$SECTION"

if [[ ! -s "$SECTION" ]]; then
  echo "Could not extract v$NEW_VERSION section from release branch changelog." >&2
  exit 1
fi

# Insert the section above the first existing heading whose version is
# numerically lower than NEW_VERSION, so the changelog stays in strictly
# descending order regardless of which patch in a family this is.
awk -v new_major="$MAJOR" -v new_minor="$MINOR" -v new_patch="$NEW_PATCH" \
    -v section_file="$SECTION" '
  BEGIN {
    while ((getline line < section_file) > 0) section = section line "\n"
    close(section_file)
    inserted = 0
  }
  !inserted && $0 ~ /^v[0-9]+\.[0-9]+\.[0-9]+/ {
    match($0, /^v[0-9]+\.[0-9]+\.[0-9]+/)
    verstr = substr($0, RSTART + 1, RLENGTH - 1)
    split(verstr, parts, ".")
    h_major = parts[1] + 0
    h_minor = parts[2] + 0
    h_patch = parts[3] + 0

    is_lower = 0
    if (h_major != new_major)      { is_lower = (h_major < new_major) }
    else if (h_minor != new_minor) { is_lower = (h_minor < new_minor) }
    else                            { is_lower = (h_patch < new_patch) }

    if (is_lower) {
      printf "%s\n", section
      inserted = 1
    }
  }
  { print }
  END {
    if (!inserted) {
      # Every existing heading is >= NEW_VERSION, or there are no headings
      # at all — append at the end.
      printf "\n%s", section
    }
  }
' "$MAIN_CHANGELOG" > "$MERGED"

cp "$MERGED" docs/changelog.rst

echo "docs/changelog.rst updated with v$NEW_VERSION." >&2
