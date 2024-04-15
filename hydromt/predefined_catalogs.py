"""Implementation of the predefined data catalogs entry points."""

import hashlib
import logging
from pathlib import Path
from typing import Dict

import yaml

from hydromt._compat import entry_points

logger = logging.getLogger(__name__)

# core artifact data first
GIT_ROOT = r"https://github.com/Deltares/hydromt/blob/main/data/catalogs"
LOCAL_EPS = {
    "artifact_data": f"{GIT_ROOT}/artifact_data/versions.yml",
    "deltares_data": f"{GIT_ROOT}/deltares_data/versions.yml",
    "aws_data": f"{GIT_ROOT}/aws_data/versions.yml",
    "gcs_cmip6_data": f"{GIT_ROOT}/gcs_cmip6_data/versions.yml",
}


def _get_catalog_eps(logger=logger) -> Dict:
    """Discover hydromt catalog plugins based on 'hydromt.catalogs' entrypoints."""
    eps = LOCAL_EPS.copy()
    for ep in entry_points(group="hydromt.catalogs"):
        name = ep.name
        if name in eps:
            plugin = f"{ep.module}.{ep.value}"
            logger.warning(f"Duplicated catalog plugin '{name}'; skipping {plugin}")
            continue

        logger.debug(f"Discovered data catalog '{name} = {ep.value}'")
        eps[ep.name] = ep.value
    return eps


def _get_file_hash(file_path: Path) -> str:
    """Get the md5 hash of a file."""
    hash_func = hashlib.md5()
    with open(file_path, "rt") as f:
        for line in f.readline():
            hash_func.update(line.encode("utf-8"))
    return hash_func.hexdigest()


def _get_catalog_versions(root):
    """Create a versions file for a catalog."""
    # discover versions based yml files in root
    versions = []
    root = Path(root)
    for f in root.glob("**/data_catalog.yml"):
        # read yml file and get hash
        with open(f, "r") as stream:
            cat_dict = yaml.safe_load(stream)
        version = cat_dict.get("meta", {}).get("version", None)
        if not version:
            raise ValueError(f"Version not found in {f}")
        hash = _get_file_hash(f)
        path = f.relative_to(root).as_posix()
        versions.append({"version": version, "hash": hash, "path": path})
    return versions
