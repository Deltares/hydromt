# -*- coding: utf-8 -*-
"""Utils for parsing cli options and arguments."""

import json
from ast import literal_eval
from os.path import isfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import click

from hydromt._io import _config_read

__all__ = ["parse_json", "parse_config", "parse_opt"]


### CLI callback methods ###
def parse_opt(ctx, param, value):
    """Parse extra cli options.

    Parse options like `--opt KEY1=VAL1 --opt SECT.KEY2=VAL2` and collect
    in a dictionary like the one below, which is what the CLI function receives.
    If no value or `None` is received then an empty dictionary is returned.
        {
            'KEY1': 'VAL1',
            'SECT': {
                'KEY2': 'VAL2'
                }
        }
    Note: `==VAL` breaks this as `str.split('=', 1)` is used.
    """
    # param and ctx are required by click though we don't use them

    out = {}
    if not value:
        return out
    for pair in value:
        if "=" not in pair:
            raise click.BadParameter("Invalid syntax for KEY=VAL arg: {}".format(pair))
        else:
            k, v = pair.split("=", 1)
            k = k.lower()
            s = None
            if "." in k:
                s, k = k.split(".", 1)
            try:
                v = literal_eval(v)
            except Exception:
                pass
            if s:
                if s not in out:
                    out[s] = dict()
                out[s].update({k: v})
            else:
                out.update({k: v})
    return out


def parse_json(ctx, param, value: str) -> Dict[str, Any]:
    """Parse json from object or file.

    If the object passed is a path pointing to a file, load it's contents and parse it.
    Otherwise attempt to parse the object as JSON itself.
    """
    # param and ctx are required by click though we don't use them
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
def parse_config(
    path: Optional[Union[Path, str]] = None, opt_cli: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Parse config from `path` and combine with command line options `opt_cli`."""
    opt = {}
    if path is not None and isfile(path):
        opt = _config_read(path, abs_path=True, skip_abspath_sections=["setup_config"])
    elif path is not None:
        raise IOError(f"Config not found at {path}")
    if opt_cli is not None:
        for section in opt_cli:
            if not isinstance(opt_cli[section], dict):
                raise ValueError(
                    "No section found in --opt values: "
                    "use <section>.<option>=<value> notation."
                )
            if section not in opt:
                opt[section] = opt_cli[section]
                continue
            for option, value in opt_cli[section].items():
                opt[section].update({option: value})
    return opt
