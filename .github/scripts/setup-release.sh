#!/usr/bin/env bash
# Bump version in source, optionally update changelog and switcher.json,
# then commit and push to the current branch.
#
# Usage: setup-release.sh <NEW_VERSION>
#
# - NEW_VERSION must be in X.Y.Z form (without leading 'v').
#
# This script does NOT create tags or PRs. Those are handled in the workflows
# so the responsibilities stay clear.

set -e

NEW_VERSION="$1"

if [[ -z "$NEW_VERSION" ]]; then
  echo "Usage: setup-release.sh <NEW_VERSION>"
  exit 1
fi

git config user.name "GitHub Actions Bot"
git config user.email "<>"

# Update version in python file.
bash .github/scripts/set-version.sh "$NEW_VERSION"
git add hydromt/__init__.py

# Update changelog header (replace Unreleased section title).
NEW_HEADER="v$NEW_VERSION ($(date -I))"
NEW_UNDERLINE=$(printf '=%.0s' $(seq 1 ${#NEW_HEADER}))
sed -i "/^Unreleased$/{s/Unreleased/$NEW_HEADER/;n;s/=*/$NEW_UNDERLINE/}" docs/changelog.rst

# Update switcher.json: insert new entry, sorted by version, with 'latest' last.
jq "map(select(.version != \"latest\")) \
  + [{\"name\":\"v$NEW_VERSION\",\"version\":\"$NEW_VERSION\",\"url\":\"https://deltares.github.io/hydromt/v$NEW_VERSION/\"}] \
  | sort_by(.version | split(\".\") | map(tonumber)) \
  + [{\"name\":\"latest\",\"version\":\"latest\",\"url\":\"https://deltares.github.io/hydromt/latest/\"}]" \
  docs/_static/switcher.json > tmp.json
mv tmp.json docs/_static/switcher.json

git add docs/changelog.rst docs/_static/switcher.json
git commit -m "Prepare release v$NEW_VERSION"
git push
