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

# Update changelog
bash .github/scripts/update-changelog.sh "$NEW_VERSION" "$RELEASE_BRANCH" "$VERSION_BUMPED"

# Update switcher.json
bash .github/scripts/update-switcher.sh "$NEW_VERSION" "$RELEASE_BRANCH" "$MARK_AS_LATEST"

# Commit and push.
git add hydromt/__init__.py docs/changelog.rst docs/_static/switcher.json
git commit -m "Record v$NEW_VERSION release on main" --allow-empty
git push --set-upstream origin "$RECORD_BRANCH"

echo "$RECORD_BRANCH"
