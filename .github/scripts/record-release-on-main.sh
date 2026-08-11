#!/usr/bin/env bash
# Merge the release branch back into main via a PR.
# Starts from the release branch so code changes (hotfixes, version bumps)
# are carried back. Pre-merges main to ensure the PR is conflict-free.
#
# Version is set to max(main's current, X.(Y+1).0.dev0).
#
# Usage: record-release-on-main.sh <RELEASE_BRANCH> <NEW_VERSION> <MARK_AS_LATEST>
#
# MARK_AS_LATEST ("true"/"false") controls whether this version is marked
# `preferred` in switcher.json — i.e. the stable release the docs switcher
# defaults to. Should be passed the same value used for
# `gh release create --latest=...` and the `stable` symlink update, so all
# three always agree. When false, existing preferred flags are left alone.
#
# Prerequisites:
#  - origin/main and release branch fetched (full history)
#  - git configured with committer identity
#  - jq and awk available
#
# Creates record-release/v<NEW_VERSION> from the release branch, merges
# origin/main (auto-resolves version/changelog/switcher conflicts, fails on
# others), sets version, rebuilds changelog and switcher, pushes the branch.
# Echoes the branch name to stdout.

set -euo pipefail

RELEASE_BRANCH="${1:-}"
NEW_VERSION="${2:-}"
MARK_AS_LATEST="${3:-false}"

if [[ -z "$RELEASE_BRANCH" || -z "$NEW_VERSION" ]]; then
  echo "Usage: record-release-on-main.sh <RELEASE_BRANCH> <NEW_VERSION> <MARK_AS_LATEST>" >&2
  exit 1
fi

RECORD_BRANCH="record-release/v$NEW_VERSION"

# Compute the target version for main (at least X.(Y+1).0.dev0).
MAJOR=$(echo "$NEW_VERSION" | cut -d. -f1)
MINOR=$(echo "$NEW_VERSION" | cut -d. -f2)
COMPUTED_NEXT="${MAJOR}.$((MINOR + 1)).0.dev0"

# Read main's current version.
VERSION_LINE=$(git show origin/main:hydromt/__init__.py | grep "^__version__")
MAIN_VERSION=$(echo "$VERSION_LINE" | cut -d= -f2 | tr -d "\"' \t")
if [[ -z "$MAIN_VERSION" ]]; then
  echo "ERROR: failed to read version from the main branch" >&2
  exit 1
fi

# Pick whichever version is higher.
MAIN_MAJOR=$(echo "$MAIN_VERSION" | cut -d. -f1)
MAIN_MINOR=$(echo "$MAIN_VERSION" | cut -d. -f2)
COMP_MAJOR=$(echo "$COMPUTED_NEXT" | cut -d. -f1)
COMP_MINOR=$(echo "$COMPUTED_NEXT" | cut -d. -f2)

if [[ "$MAIN_MAJOR" -gt "$COMP_MAJOR" ]] || \
   { [[ "$MAIN_MAJOR" -eq "$COMP_MAJOR" ]] && [[ "$MAIN_MINOR" -ge "$COMP_MINOR" ]]; }; then
  TARGET_VERSION="$MAIN_VERSION"
else
  TARGET_VERSION="$COMPUTED_NEXT"
fi

# Create branch from the release branch.
git checkout -B "$RECORD_BRANCH" "origin/$RELEASE_BRANCH"

# Merge main into the branch. Metadata files are rebuilt after merge.
MERGE_FAILED=false
git merge origin/main --no-commit --no-edit 2>/dev/null || MERGE_FAILED=true

if [[ "$MERGE_FAILED" == "true" ]]; then
  # Auto-resolve conflicts in files we rebuild anyway.
  KNOWN_FILES="hydromt/__init__.py docs/changelog.rst docs/_static/switcher.json"
  for f in $KNOWN_FILES; do
    if git diff --name-only --diff-filter=U | grep -qx "$f"; then
      git checkout --theirs "$f" 2>/dev/null || true
      git add "$f"
    fi
  done

  # Fail on remaining unresolved conflicts.
  REMAINING=$(git diff --name-only --diff-filter=U || true)
  if [[ -n "$REMAINING" ]]; then
    echo "ERROR: Unresolved merge conflicts in the following files:" >&2
    echo "$REMAINING" >&2
    echo "Please resolve these conflicts manually on the release branch" >&2
    echo "(e.g. via cherry-pick) before retrying." >&2
    git merge --abort
    exit 1
  fi
fi

# Complete the merge commit when the merge created a pending merge state.
# If origin/main is already up to date, `git merge --no-commit` does not
# leave MERGE_HEAD and there is nothing to commit.
if git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
  git commit --no-edit -m "Merge main into record-release/v$NEW_VERSION" \
    --allow-empty 2>/dev/null
fi

# Set correct version.
VERSION_BUMPED=false
if [[ "$TARGET_VERSION" != "$MAIN_VERSION" ]]; then
  VERSION_BUMPED=true
fi
bash .github/scripts/set-version.sh "$TARGET_VERSION"

# Rebuild changelog from main's copy.
git show origin/main:docs/changelog.rst > /tmp/main-changelog.rst

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
  ' /tmp/main-changelog.rst > /tmp/main-changelog-fresh.rst
  mv /tmp/main-changelog-fresh.rst /tmp/main-changelog.rst
fi

# Extract the v<NEW_VERSION> section from the release branch changelog.
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

# Insert the section above the first existing heading whose version is
# numerically lower than NEW_VERSION, so the changelog stays in strictly
# descending order regardless of which patch in a family this is.
NEW_PATCH=$(echo "$NEW_VERSION" | cut -d. -f3)

awk -v new_major="$MAJOR" -v new_minor="$MINOR" -v new_patch="$NEW_PATCH" \
    -v section_file="/tmp/section.rst" '
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
' /tmp/main-changelog.rst > /tmp/changelog-merged.rst

cp /tmp/changelog-merged.rst docs/changelog.rst

# Rebuild switcher.json: union of both sides.
git show origin/main:docs/_static/switcher.json > /tmp/main-switcher.json
git show "origin/$RELEASE_BRANCH:docs/_static/switcher.json" > /tmp/release-switcher.json
jq -s '
  (.[0] + .[1])
  | map(select(.version != "latest"))
  | unique_by(.version)
  | sort_by(.version | split(".") | map(tonumber? // 0))
  + [{"name":"latest","version":"latest","url":"https://deltares.github.io/hydromt/latest/"}]
' /tmp/main-switcher.json /tmp/release-switcher.json > /tmp/switcher-merged.json

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
  ' /tmp/switcher-merged.json > /tmp/switcher-preferred.json

  PREFERRED_COUNT=$(jq '[.[] | select(.version != "latest" and .preferred == true)] | length' /tmp/switcher-preferred.json)
  if [[ "$PREFERRED_COUNT" -ne 1 ]]; then
    echo "ERROR: switcher.json has $PREFERRED_COUNT preferred entries, expected exactly 1." >&2
    exit 1
  fi

  mv /tmp/switcher-preferred.json /tmp/switcher-merged.json
fi


cp /tmp/switcher-merged.json docs/_static/switcher.json

# Commit and push.
git add hydromt/__init__.py docs/changelog.rst docs/_static/switcher.json
git commit -m "Record v$NEW_VERSION release on main" --allow-empty
git push --set-upstream origin "$RECORD_BRANCH"

echo "$RECORD_BRANCH"
