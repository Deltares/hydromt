#!/usr/bin/env bash
# Reads the current version from hydromt/__init__.py and prints it to stdout.
# Usage: get-version.sh

grep "__version__" hydromt/__init__.py | cut -d= -f 2 | tr -d "\" "
