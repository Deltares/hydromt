"""config I/O functions"""

import abc
from ast import literal_eval
import codecs
from configparser import ConfigParser
from os.path import dirname, join, abspath, exists, splitext
from pathlib import Path
from typing import Union, Dict, List
import yaml

__all__ = [
    "configread",
    "configwrite",
]


def configread(
    config_fn: Union[Path, str],
    defaults: Dict = dict(),
    abs_path: bool = False,
    skip_abspath_sections: List = ["setup_config"],
    **kwargs,
) -> Dict:
    """Read configuration/workflow file and parse to (nested) dictionary.

    Parameters
    ----------
    config_fn : Union[Path, str]
        Path to configuration file
    defaults : dict, optional
        Nested dictionary with default options, by default dict()
    abs_path : bool, optional
        If True, parse string values to an absolute path if the a file or folder with that
        name (string value) relative to the config file exist, by default False
    skip_abspath_sections: list, optional
        These sections are not evaluated for absolute paths if abs_path=True,
        by default ['update_config']

    Returns
    -------
    cfdict : dict
        Configuration dictionary.
    """
    # read
    if splitext(config_fn)[-1] in [".yaml", ".yml"]:
        with open(config_fn, "rb") as f:
            cfdict = yaml.safe_load(f)
    else:
        cfdict = read_ini_config(config_fn, **kwargs)
    # parse absolute paths
    if abs_path:
        root = Path(dirname(config_fn))
        cfdict = parse_abspath(cfdict, root, skip_abspath_sections)
    # update defaults
    if defaults:
        _cfdict = defaults.copy()
        _cfdict.update(cfdict)
        cfdict = _cfdict
    return cfdict


def configwrite(config_fn: Union[str, Path], cfdict: dict, **kwargs) -> None:
    """Write configuration/workflow dictionary to file.

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
    root = Path(dirname(config_fn))
    _cfdict = parse_relpath(cfdict.copy(), root)
    if splitext(config_fn)[-1] in [".yaml", ".yml"]:
        with open(config_fn, "w") as f:
            yaml.dump(_cfdict, f, sort_keys=False)
    else:
        write_ini_config(config_fn, _cfdict, **kwargs)


def read_ini_config(
    config_fn: Union[Path, str],
    encoding: str = "utf-8",
    cf: ConfigParser = None,
    skip_eval: bool = False,
    skip_eval_sections: list = [],
    noheader: bool = False,
) -> dict:
    """Read configuration ini file and parse to (nested) dictionary.

    Parameters
    ----------
    config_fn : Union[Path, str]
        Path to configuration file
    encoding : str, optional
        File encoding, by default "utf-8"
    cf : ConfigParser, optional
        Alternative configuration parser, by default None
    skip_eval : bool, optional
        If True, do not evaluate string values, by default False
    skip_eval_sections : list, optional
        These sections are not evaluated for string values
        if skip_eval=True, by default []
    noheader : bool, optional
        Set true for a single-level configuration file with no headers, by default False

    Returns
    -------
    cfdict : dict
        Configuration dictionary.
    """
    if cf is None:
        cf = ConfigParser(allow_no_value=True, inline_comment_prefixes=[";", "#"])
    elif isinstance(cf, abc.ABCMeta):  # not yet instantiated
        cf = cf()
    cf.optionxform = str  # preserve capital letter
    with codecs.open(config_fn, "r", encoding=encoding) as fp:
        cf.read_file(fp)
        cfdict = cf._sections
    # parse values
    cfdict = parse_values(cfdict, skip_eval, skip_eval_sections)
    # add dummy header
    if noheader and "dummy" in cfdict:
        cfdict = cfdict["dummy"]
    return cfdict


def write_ini_config(
    config_fn: Union[Path, str],
    cfdict: dict,
    encoding: str = "utf-8",
    cf: ConfigParser = None,
    noheader: bool = False,
) -> None:
    """Write configuration dictionary to ini file.

    Parameters
    ----------
    config_fn : Union[Path, str]
        Path to configuration file
    cfdict : dict
        Configuration dictionary.
    encoding : str, optional
        File encoding, by default "utf-8"
    cf : ConfigParser, optional
        Alternative configuration parser, by default None
    noheader : bool, optional
        Set true for a single-level configuration dictionary with no headers, by default False
    """
    if cf is None:
        cf = ConfigParser(allow_no_value=True, inline_comment_prefixes=[";", "#"])
    elif isinstance(cf, abc.ABCMeta):  # not yet instantiated
        cf = cf()
    cf.optionxform = str  # preserve capital letter
    if noheader:  # add dummy header
        cfdict = {"dummy": cfdict}
    cf.read_dict(cfdict)
    with codecs.open(config_fn, "w", encoding=encoding) as fp:
        cf.write(fp)


def parse_relpath(cfdict: dict, root: Path) -> dict:
    """Parse string/path value to relative path if possible."""

    def _relpath(value, root):
        if isinstance(value, str) and str(Path(value)).startswith(str(root)):
            value = Path(value)
        if isinstance(value, Path):
            try:
                rel_path = value.relative_to(root)
                value = str(rel_path).replace("\\", "/")
            except ValueError:
                pass  # `value` path is not relative to root
        return value

    # loop through n-level of dict
    for key, val in cfdict.items():
        if isinstance(val, dict):
            cfdict[key] = parse_relpath(val, root)
        else:
            cfdict[key] = _relpath(val, root)
    return cfdict


def parse_abspath(
    cfdict: dict, root: Path, skip_abspath_sections: List = ["setup_config"]
) -> dict:
    """Parse string value to absolute path if the a file or folder with that
    name (string value) relative to the config file exist.
    """

    def _abspath(value, root):
        if isinstance(value, str) and exists(join(root, value)):
            value = Path(abspath(join(root, value)))
        elif (
            isinstance(value, list)
            and all([isinstance(v, str) for v in value])
            and all([exists(join(root, v)) for v in value])
        ):
            value = [Path(abspath(join(root, v))) for v in value]
        return value

    # loop through n-level of dict
    for key, val in cfdict.items():
        if isinstance(val, dict):
            if key not in skip_abspath_sections:
                cfdict[key] = parse_abspath(val, root)
        else:
            cfdict[key] = _abspath(val, root)
    return cfdict


def parse_values(
    cfdict: dict,
    skip_eval: bool = False,
    skip_eval_sections: List = [],
):
    """Parse string values to python default objects.

    Parameters
    ----------
    cfdict : dict
        Configuration dictionary.
    skip_eval : bool, optional
        Set true to skip evaluation, by default False
    skip_eval_sections : List, optional
        List of sections to skip evaluation, by default []

    Returns
    -------
    cfdict : dict
        Configuration dictionary with evaluated values.
    """
    # loop through two-level dict: section, key-value pairs
    for section in cfdict:
        # evaluate ini items to parse to python default objects:
        if skip_eval or section in skip_eval_sections:
            cfdict[section].update(
                {key: str(var) for key, var in cfdict[section].items()}
            )  # cast None type values to str
            continue  # do not evaluate
        # numbers, tuples, lists, dicts, sets, booleans, and None
        for key, value in cfdict[section].items():
            try:
                value = literal_eval(value)
            except Exception:
                pass
            if isinstance(value, str) and len(value) == 0:
                value = None
            cfdict[section].update({key: value})
    return cfdict
