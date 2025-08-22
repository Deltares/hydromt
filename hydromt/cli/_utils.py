# -*- coding: utf-8 -*-
"""Utils for parsing cli options and arguments."""

import json
from os.path import isfile
from pathlib import Path
from typing import Any, Dict, Union

import click

from hydromt._io import _config_read

__all__ = ["parse_json", "parse_config"]


### CLI callback methods ###
def parse_json(_ctx: click.Context, _param, value: str) -> Dict[str, Any]:
    """Parse json from object or file.

    If the object passed is a path pointing to a file, load it's contents and parse it.
    Otherwise attempt to parse the object as JSON itself.
    """
    if isfile(value):
        with open(value, "r") as f:
            kwargs = json.load(f)
    else:
        if value.strip("{").startswith("'"):
            value = value.replace("'", '"')
        try:
            kwargs = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(f'Could not decode JSON "{value}"')
    return kwargs


### general parsing methods ##
def parse_config(path: Union[Path, str]) -> Dict[str, Any]:
    """Parse config from `path`."""
    if not isfile(path):
        raise IOError(f"Config not found at {path}")

    return _config_read(path, abs_path=True, skip_abspath_sections=["setup_config"])
