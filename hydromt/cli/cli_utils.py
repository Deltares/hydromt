# -*- coding: utf-8 -*-
"""Utils for parsing cli options and arguments."""

import json
import logging
import os
from ast import literal_eval
from os.path import isfile
from pathlib import Path
from typing import Any, Dict, Union
from warnings import warn

import click
import requests

from .. import config
from ..error import DeprecatedError

logger = logging.getLogger(__name__)

__all__ = ["parse_json", "parse_config", "parse_opt", "download_examples"]

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
    if isfile(value):
        with open(value, "r") as f:
            kwargs = json.load(f)

    # Catch old keyword for resulution "-r"
    elif type(literal_eval(value)) in (float, int):
        raise DeprecatedError("'-r' is used for region, resolution is deprecated")
    else:
        if value.strip("{").startswith("'"):
            value = value.replace("'", '"')
        try:
            kwargs = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(f'Could not decode JSON "{value}"')
    return kwargs


### general parsing methods ##


def parse_config(path: Union[Path, str] = None, opt_cli: Dict = None) -> Dict:
    """Parse config from ini `path` and combine with command line options `opt_cli`."""
    opt = {}
    if path is not None and isfile(path):
        if str(path).endswith(".ini"):
            warn(
                "Support for .ini configuration files will be deprecated",
                PendingDeprecationWarning,
                stacklevel=2,
            )
        opt = config.configread(
            path, abs_path=True, skip_abspath_sections=["setup_config"]
        )
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


def download_examples(
    examples_path: Path,
    examples_out: Path,
    examples_path_raw: Path = None,
    logger: logging.Logger = logger,
):
    """
    Discover files in examples_path and download them to examples_out.

    Parameters
    ----------
    examples_path : str
        URL to the examples directory on GitHub for discovery.
    examples_out : str
        Local path to the examples directory.
    examples_path_raw : str, optional
        URL to the raw examples directory on GitHub if different than examples_path.
    """

    def download_file(url, destination_path):
        response = requests.get(url)
        with open(destination_path, "wb") as f:
            f.write(response.content)

    def download_folder(examples_path, examples_out, examples_path_raw, logger):
        try:
            # Use requests to get the contents of the directory
            response = requests.get(examples_path)
            contents = response.json()

            # GitHub https contains a lot of info... we only need the tree items
            contents = contents["payload"]["tree"]["items"]

            # Create the directory structure
            for item in contents:
                # If subdirectory in examples, check for files in them
                if item["contentType"] == "directory":
                    os.makedirs(os.path.join(examples_out, item["name"]), exist_ok=True)

                    # Recursively download the contents of the subdirectory
                    download_folder(
                        examples_path + "/" + item["name"],
                        os.path.join(examples_out, item["name"]),
                        examples_path_raw + "/" + item["name"],
                        logger,
                    )
                else:
                    logger.debug(f"Downloading {item['name']}")
                    download_file(
                        examples_path_raw + "/" + item["name"],
                        os.path.join(examples_out, item["name"]),
                    )
        except Exception as e:
            logger.error(f"Failed to download folder {examples_path}: {e}")

    if examples_path_raw is None:
        examples_path_raw = examples_path

    # Create the output directory
    os.makedirs(examples_out, exist_ok=True)
    # Download the contents of the examples directory
    download_folder(examples_path, examples_out, examples_path_raw, logger)
