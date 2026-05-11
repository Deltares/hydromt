#!/usr/bin/env bash
# Merge the release branch back into main via a PR.
# Starts from the release branch so code changes (hotfixes, version bumps)
# are carried back. Pre-merges main to ensure the PR is conflict-free.
#
# Version is set to max(main's current, X.(Y+1).0.dev0).
#
# Usage: record-release-on-main.sh <RELEASE_BRANCH> <NEW_VERSION>
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

if [[ -z "$RELEASE_BRANCH" || -z "$NEW_VERSION" ]]; then
  echo "Usage: record-release-on-main.sh <RELEASE_BRANCH> <NEW_VERSION>" >&2
  exit 1
fi

RECORD_BRANCH="record-release/v$NEW_VERSION"

# Compute the target version for main (at least X.(Y+1).0.dev0).
MAJOR=$(echo "$NEW_VERSION" | cut -d. -f1)
MINOR=$(echo "$NEW_VERSION" | cut -d. -f2)
COMPUTED_NEXT="${MAJOR}.$((MINOR + 1)).0.dev0"

# Read main's current version.
MAIN_VERSION=$(git show origin/main:hydromt/__init__.py \
  | grep "__version__" | cut -d= -f2 | tr -d "\" ")

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

# Complete the merge commit (--no-commit requires explicit commit).
git commit --no-edit -m "Merge main into record-release/v$NEW_VERSION" \
  --allow-empty 2>/dev/null || true

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

# Insert the section above the matching X.Y.0 heading in main's changelog.
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
  !inserted && /^v[0-9]+\.[0-9]+\.[0-9]+/ {
    # family_base not in changelog; insert before the first version heading.
    printf "%s\n", section
    inserted = 1
  }
  { print }
  END {
    if (!inserted) {
      # No version headings at all; append.
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
cp /tmp/switcher-merged.json docs/_static/switcher.json

# Commit and push.
git add hydromt/__init__.py docs/changelog.rst docs/_static/switcher.json
git commit -m "Record v$NEW_VERSION release on main" --allow-empty
git push --set-upstream origin "$RECORD_BRANCH"

echo "$RECORD_BRANCH"
