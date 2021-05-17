#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config functions
"""

import codecs
from configparser import ConfigParser
from os.path import dirname, join, abspath, exists
from ast import literal_eval
import numpy as np
import abc
from pathlib import Path


def configread(
    config_fn,
    encoding="utf-8",
    cf=None,
    defaults=dict(),
    noheader=False,
    abs_path=False,
):
    """read model configuration from file and parse to dictionary"""
    if cf is None:
        cf = ConfigParser(allow_no_value=True, inline_comment_prefixes=[";", "#"])
    elif isinstance(cf, abc.ABCMeta):  # not yet instantiated
        cf = cf()
    cf.optionxform = str  # preserve capital letter
    with codecs.open(config_fn, "r", encoding=encoding) as fp:
        cf.read_file(fp)
    root = dirname(config_fn)
    cfdict = defaults.copy()
    for section in cf.sections():
        if section not in cfdict:
            cfdict[section] = dict()  # init
        sdict = dict()
        for key, value in cf.items(section):
            try:
                v = literal_eval(value)
                assert not isinstance(v, tuple)  #  prevent tuples from being parsed
                value = v
            except Exception:
                pass
            if abs_path:
                if isinstance(value, str) and exists(join(root, value)):
                    value = Path(abspath(join(root, value)))
                elif isinstance(value, list) and np.all(
                    [exists(join(root, v)) for v in value]
                ):
                    value = [Path(abspath(join(root, v))) for v in value]
            sdict[key] = value
        cfdict[section].update(**sdict)
    if noheader and "dummy" in cfdict:
        cfdict = cfdict["dummy"]
    return cfdict


def configwrite(config_fn, cfdict, encoding="utf-8", cf=None, noheader=False):
    """write model configuration to file"""
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
