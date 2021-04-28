# -*- coding: utf-8 -*-
"""Utils for parsing cli options and arguments 
"""

from os.path import join, isfile, isdir
import numpy as np
import json
import geopandas as gpd
import logging
import click
from ast import literal_eval

from .. import config

logger = logging.getLogger(__name__)

__all__ = ["parse_json", "parse_config", "parse_opt"]

### CLI callback methods ###


def parse_opt(ctx, param, value):
    """
    click callback to validate `--opt KEY1=VAL1 --opt SECT.KEY2=VAL2` and collect
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


def parse_json(ctx, param, value):
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


### general parsin methods ##


def parse_config(path=None, opt_cli=None, components=None, logger=logger):
    opt = {}
    if path is not None and isfile(path):
        opt = config.configread(path, abs_path=True)
        # make sure paths in config section are not abs paths
        if "setup_config" in opt:
            opt["setup_config"].update(config.configread(path).get("config", {}))
    elif path is not None:
        raise IOError(f"Config not found at {path}")
    if opt_cli is not None:
        for section in opt_cli:
            if not isinstance(opt_cli[section], dict):
                raise ValueError(
                    f"No section found in --opt values: "
                    "use <section>.<option>=<value> notation."
                )
            if section not in opt:
                opt[section] = opt_cli[section]
                continue
            for option, value in opt_cli[section].items():
                opt[section].update({option: value})
    for section in opt:
        for option in opt[section]:
            value = opt[section][option]
            if logger is not None and (components is None or section in components):
                logger.info(f"{section}.{option}: {value}")
    return opt
