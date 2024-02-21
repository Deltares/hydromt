"""A module to handle paths from different platforms in a cross platform compatible manner."""
from os.path import abspath, exists, join
from pathlib import Path
from typing import List, Optional


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
    cfdict: dict, root: Path, skip_abspath_sections: Optional[List] = None
) -> dict:
    """Parse string value to absolute path from config file."""
    skip_abspath_sections = skip_abspath_sections or ["setup_config"]

    def _abspath(value, root):
        if exists(join(root, value)):
            value = Path(abspath(join(root, value)))
        return value

    # loop through n-level of dict
    for key, val in cfdict.items():
        if isinstance(val, dict):
            if key not in skip_abspath_sections:
                cfdict[key] = parse_abspath(val, root)
        elif isinstance(val, list) and all([isinstance(v, str) for v in val]):
            cfdict[key] = [_abspath(v, root) for v in val]
        elif isinstance(val, str):
            cfdict[key] = _abspath(val, root)
    return cfdict
