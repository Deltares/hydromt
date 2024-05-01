import re
from urllib.parse import urlparse

from hydromt._typing import StrPath


def strip_scheme(uri: str) -> str:
    """Strip scheme from uri."""
    return re.sub(r"^\w+://", "", uri)


def is_valid_url(uri: StrPath) -> bool:
    """Check if uri is valid."""
    try:
        result = urlparse(str(uri))
        return all([result.scheme, result.netloc])
    except (ValueError, AttributeError):
        return False
