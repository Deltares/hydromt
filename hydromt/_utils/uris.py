import re
from urllib.parse import urlparse

from hydromt._typing import StrPath

__all__ = ["_strip_scheme", "_is_valid_url"]


def _strip_scheme(uri: str) -> str:
    """Strip scheme from uri."""
    return re.sub(r"^\w+://", "", uri)


def _is_valid_url(uri: StrPath) -> bool:
    """Check if uri is valid."""
    try:
        result = urlparse(str(uri))
        return all([result.scheme, result.netloc])
    except (ValueError, AttributeError):
        return False
