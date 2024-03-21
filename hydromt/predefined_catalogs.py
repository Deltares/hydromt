"""Implementation of the predefined data catalogs entry points."""

import logging
from typing import Dict

from hydromt._compat import entry_points

logger = logging.getLogger(__name__)

# core artifact data first
LOCAL_EPS = {
    "artifact_data": r"https://github.com/Deltares/hydromt/blob/main/data/catalogs/artifact_data/versions.yml",
    "deltares_data": r"https://github.com/Deltares/hydromt/blob/main/data/catalogs/deltares_data/versions.yml",
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
