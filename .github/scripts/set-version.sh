#!/usr/bin/env bash
# Set the current version in hydromt/__init__.py
# Usage: set-version.sh <NEW_VERSION>

export NEW_VERSION=$1

sed -i "s/.*__version__.*/__version__ = \"$NEW_VERSION\"/" hydromt/__init__.py
