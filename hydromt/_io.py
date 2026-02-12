"""low-level IO functions for Hydromt, should not depend on other hydromt modules."""

import sys
from pathlib import Path
from typing import Any

import requests
import yaml

from hydromt._utils.uris import _is_valid_url

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = ["read_yaml", "read_toml", "read_uri", "yml_from_uri_or_path"]


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read yaml file and return as dict."""
    with open(path, "rb") as stream:
        yml = yaml.safe_load(stream)
    return yml


def read_toml(path: str | Path) -> dict[str, Any]:
    """Read toml file and return as dict."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data


def read_uri(uri: str | Path) -> str:
    """Read content from a URI and return as string."""
    with requests.get(str(uri), stream=True) as r:
        r.raise_for_status()
        content = r.text
    return content


def yml_from_uri_or_path(uri_or_path: str | Path) -> dict[str, Any]:
    """Read YAML content from a URI or local path and return as dict."""
    if _is_valid_url(str(uri_or_path)):
        yml_text = read_uri(uri_or_path)
        yml = yaml.safe_load(yml_text)
    else:
        yml = read_yaml(uri_or_path)
    return yml
