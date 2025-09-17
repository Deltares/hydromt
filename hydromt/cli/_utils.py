# -*- coding: utf-8 -*-
"""Utils for parsing cli options and arguments."""

import json
from os.path import isfile
from typing import Any, Dict

import click

__all__ = ["parse_json"]


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
