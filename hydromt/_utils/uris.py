import re
from typing import Optional, Tuple
from urllib.parse import urlparse

from hydromt._typing import StrPath

__all__ = ["_strip_scheme", "_is_valid_url"]


def _strip_scheme(uri: str) -> Tuple[Optional[str], str]:
    """Strip scheme from uri."""
    try:
        scheme: str = next(re.finditer(r"^\w+://", uri)).group()
    except StopIteration:
        # no scheme found
        return (None, uri)
    return (scheme, uri.lstrip(scheme))


def _strip_vsi(uri: str) -> Tuple[Optional[str], str]:
    """Strip gdal virtual filesystem prefix."""
    try:
        prefix: str = next(re.finditer(r"^/vsi\w+/", uri)).group()
    except StopIteration:
        # No prefix found
        return None, uri
    return (prefix, uri.lstrip(prefix))


def _is_valid_url(uri: StrPath) -> bool:
    """Check if uri is valid."""
    try:
        result = urlparse(str(uri))
        return all([result.scheme, result.netloc])
    except (ValueError, AttributeError):
        return False
