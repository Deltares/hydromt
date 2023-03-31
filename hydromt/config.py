#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config functions
"""
from os.path import dirname, join, abspath, exists, splitext
from pathlib import Path
from typing import Union, Dict, List
from ast import literal_eval
import codecs
import abc
from configparser import ConfigParser
from warnings import warn

import tomli
import yaml


def configread(
    config_fn: Union[Path, str],
    encoding: str = "utf-8",
    cf: ConfigParser = None,
    defaults: Dict = dict(),
    noheader: bool = False,
    abs_path: bool = False,
    skip_eval: bool = False,
    skip_eval_sections: List = [],
    skip_abspath_sections: List = ["setup_config"],
) -> Dict:
    """Read configuration file and parse to (nested) dictionary.
    Values are evaluated and if possible parsed into python int, float, list or boolean types.

    Parameters
    ----------
    config_fn : Union[Path, str]
        Path to configuration file
    encoding : str, optional
        File encoding, by default "utf-8"
    cf : ConfigParser, optional
        Alternative configuration parser, by default None
    defaults : dict, optional
        Nested dictionary with default options, by default dict()
    noheader : bool, optional
        Set true for a single-level configuration file with no headers, by default False
    abs_path : bool, optional
        If True, parse string values to an absolute path if the a file or folder with that
        name (string value) relative to the config file exist, by default False
    skip_eval: bool, optional
        Skip evaluating argument values for python types, by default False
    skip_eval_sections: list, optional
        These sections are not evaluated for python types or absolute paths
        if abs_path=True, by default []
    skip_abspath_sections: list, optional
        These sections are not evaluated for absolute paths
        if abs_path=True, by default ['update_config']

    Returns
    -------
    cfdict : dict
        Configuration dictionary. If the configuration contains headers,
        the first level keys are the section headers, the second level option-value pairs.
    """
    if splitext(config_fn)[-1] in [".yaml", ".yml"]:
        cf = parse_yaml_config(config_fn)

    else:
        warn(
            "Support for .ini configuration files will be deprecated",
            PendingDeprecationWarning,
        )
        if cf is None:
            cf = ConfigParser(allow_no_value=True, inline_comment_prefixes=[";", "#"])
        elif isinstance(cf, abc.ABCMeta):  # not yet instantiated
            cf = cf()
        cf.optionxform = str  # preserve capital letter
        with codecs.open(config_fn, "r", encoding=encoding) as fp:
            cf.read_file(fp)
            cf = cf._sections
    root = dirname(config_fn)
    cfdict = defaults.copy()
    for section in cf:
        if section not in cfdict:
            cfdict[section] = dict()  # init
        # evaluate ini items to parse to python default objects:
        if skip_eval or section in skip_eval_sections:
            cfdict[section].update(
                {key: str(var) for key, var in cf[section].items()}
            )  # cast None type values to str
            continue  # do not evaluate
        # numbers, tuples, lists, dicts, sets, booleans, and None
        for key, value in cf[section].items():
            try:
                value = literal_eval(value)
            except Exception:
                pass
            if isinstance(value, str) and len(value) == 0:
                value = None
            if abs_path and section not in skip_abspath_sections:
                if isinstance(value, str) and exists(join(root, value)):
                    value = Path(abspath(join(root, value)))
                elif (
                    isinstance(value, list)
                    and all([isinstance(v, str) for v in value])
                    and all([exists(join(root, v)) for v in value])
                ):
                    value = [Path(abspath(join(root, v))) for v in value]
            cfdict[section].update({key: value})
    if noheader and "dummy" in cfdict:
        cfdict = cfdict["dummy"]
    return cfdict


def configwrite(
    config_fn: Union[str, Path],
    cfdict: dict,
    encoding: str = "utf-8",
    cf: ConfigParser = None,
    noheader: bool = False,
) -> None:
    """_summary_

    Parameters
    ----------
    config_fn : Union[Path, str]
        Path to configuration file
    cfdict : dict
        Configuration dictionary. If the configuration contains headers,
        the first level keys are the section headers, the second level option-value pairs.
    encoding : str, optional
        File encoding, by default "utf-8"
    cf : ConfigParser, optional
        Alternative configuration parser, by default None
    noheader : bool, optional
        Set true for a single-level configuration dictionary with no headers, by default False
    """
    _cfdict = cfdict.copy()
    root = Path(dirname(config_fn))
    if cf is None:
        cf = ConfigParser(allow_no_value=True, inline_comment_prefixes=[";", "#"])
    elif isinstance(cf, abc.ABCMeta):  # not yet instantiated
        cf = cf()
    if noheader:
        _cfdict = {"dummy": _cfdict}
    cf.optionxform = str  # preserve capital letter
    for sect in _cfdict.keys():
        for key, value in _cfdict[sect].items():
            if isinstance(value, str) and str(Path(value)).startswith(str(root)):
                _cfdict[sect][key] = Path(value)
            if isinstance(value, Path):
                try:
                    rel_path = value.relative_to(root)
                    _cfdict[sect][key] = str(rel_path).replace("\\", "/")
                except ValueError:
                    pass  # `value` path is not relative to root
    cf.read_dict(_cfdict)
    with codecs.open(config_fn, "w", encoding=encoding) as fp:
        cf.write(fp)


def parse_yaml_config(config_fn: Union[Path, str]) -> dict:
    with open(config_fn, "rb") as f:
        return yaml.safe_load(f)
