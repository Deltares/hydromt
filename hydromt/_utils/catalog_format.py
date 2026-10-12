"""Data catalog format versions."""

import logging
from enum import StrEnum

from hydromt import __version__

__all__ = ["LATEST_CATALOG_FORMAT", "catalog_format", "check_catalog_format", "Format"]

logger = logging.getLogger(__name__)

LATEST_CATALOG_FORMAT = 2
"""Newest data catalog format, set with ``meta.format_version``."""

OLDEST_RUNTIME_CATALOG_FORMAT = 1
"""Oldest format that can still be read (upgraded in memory) at runtime."""


class Format(StrEnum):
    v0 = "v0"
    v1 = "v1"
    v2 = "v2"


def catalog_format(format_version: str | int) -> int:
    """Return the catalog format from ``meta.format_version`` (2 or "v2").

    Parameters
    ----------
    format_version: str | int
        The catalog format version, either as an integer (e.g., 2) or a string (e.g., "v2").

    Returns
    -------
    int
        The catalog format as an integer.

    Raises
    ------
    ValueError
        If the format_version is not a valid integer or string representation of an integer.
    """
    try:
        return int(str(format_version).lstrip("vV"))
    except ValueError:
        raise ValueError(
            f"Invalid catalog format_version {format_version!r}, expected an integer."
        ) from None


def check_catalog_format(
    format_version: str | int | None, source: str = "data catalog"
) -> int:
    """Raise if the catalog format given by ``format_version`` can not be read.

    Parameters
    ----------
    format_version: str | int | None
        The catalog format version, either as an integer (e.g., 2) or a string (e.g., "v2").
        If None, a warning is issued and the v1 format is assumed.
    source: str, optional
        The name of the source for logging purposes. Default is "data catalog".

    Returns
    -------
    int
        The catalog format as an integer.

    Raises
    ------
    RuntimeError
        If the catalog format is not supported by the current version of hydromt.
    """
    if format_version is None:
        logger.warning(f"{source} does not specify a catalog format. Assuming v1.")
        return 1
    fmt = catalog_format(format_version)
    if fmt > LATEST_CATALOG_FORMAT:
        raise RuntimeError(
            f"{source} uses catalog format v{fmt}, which is "
            f"not supported by hydromt {__version__} "
            f"(supports up to v{LATEST_CATALOG_FORMAT}). "
            "Upgrade hydromt or use an older version of this catalog."
        )
    if fmt < OLDEST_RUNTIME_CATALOG_FORMAT:
        raise RuntimeError(
            f"{source} uses the legacy catalog format v{fmt}, "
            f"which hydromt {__version__} can not read. Upgrade it with "
            "`hydromt check -d <catalog.yml> --upgrade`."
        )
    return fmt
