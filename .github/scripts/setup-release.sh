#!/usr/bin/env bash
# Given a version and release branch, it updates version strings across the project,
# updates the changelog and switcher.json, commits and pushes, creates a PR, and
# comments a pyproject.toml diff since the last tag for conda receipt tracking.
#
# Usage: setup-release.sh <NEW_VERSION> <RELEASE_BRANCH>
# Expects GH_TOKEN to be set in the environment.

set -e

export NEW_VERSION=$1
export RELEASE_BRANCH=$2

if [[ -z "$NEW_VERSION" || -z "$RELEASE_BRANCH" ]]; then
  echo "Usage: setup-release.sh <NEW_VERSION> <RELEASE_BRANCH>"
  exit 1
fi

# update version in python file
sed -i "s/.*__version__.*/__version__ = \"$NEW_VERSION\"/" hydromt/__init__.py

# update changelog with new header
export NEW_CHANGELOG_HEADER="v$NEW_VERSION ($(date -I))"
export NEW_HEADER_UNDERLINE=$(printf '=%.0s' $(seq 1 $(echo $NEW_CHANGELOG_HEADER | awk '{print length}')))
sed -i "/Unreleased/{s/Unreleased/$NEW_CHANGELOG_HEADER/;n;s/=*/$NEW_HEADER_UNDERLINE/}" docs/changelog.rst

# update switcher.json
cat docs/_static/switcher.json | jq "map(select(.version != \"latest\")) | . + [{\"name\":\"v$NEW_VERSION\",\"version\":\"$NEW_VERSION\",\"url\":\"https://deltares.github.io/hydromt/v$NEW_VERSION/\"}] | sort_by(.version|split(\".\")|map(tonumber)) | . + [{\"name\":\"latest\",\"version\":\"latest\",\"url\":\"https://deltares.github.io/hydromt/latest/\"}]" > tmp.json
mv tmp.json docs/_static/switcher.json

git add .
git commit -m "prepare for release v$NEW_VERSION"
git push

export PR_URL=$(gh pr create -B "$RELEASE_BRANCH" -H "$RELEASE_BRANCH" -t "Release v$NEW_VERSION" -b "Hi there! This is an automated PR made for your release. Pushing commits to this branch will start the creation and testing of release artifacts. Due to limitations with github actions I can't start these for you but you can do so by adding an empty commit with \`git commit --allow-empty\`. Once the release artifacts are ready I'll comment links to them so you can inspect them. Please check that the PR is as you would like it to be, and feel free to push any changes to this branch before merging. Once you're happy with the state of the PR, just merge the PR and run the \`.github/workflows/finish-release.yml\` workflow.")

# not fetched by default
git fetch --tags
export LAST_RELEASE=$(git describe --tags --abbrev=0)
export PYPROJ_DIFF=$(git diff "$LAST_RELEASE" -- pyproject.toml)

if [[ -z $PYPROJ_DIFF ]]; then
  gh pr comment "$PR_URL" --body "I've detected the latest tag was \`$LAST_RELEASE\`. No dependency or project config changes have been made since then. You should be able to release to conda without updating the receipt."
else
  echo "I've detected the latest tag was \`$LAST_RELEASE\`. Here are the changes made to the \`pyproject.toml\` since then:" > comment.txt
  echo $'\n\n```diff\n' >> comment.txt
  echo "$PYPROJ_DIFF" >> comment.txt
  echo $'\n\n```\n' >> comment.txt
  echo "(you might have to update the receipt to update to reflect these changes when releasing to conda)"
  gh pr comment "$PR_URL" --body-file comment.txt
fi
