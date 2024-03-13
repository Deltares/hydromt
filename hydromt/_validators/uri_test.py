# test whether something is a uri
import logging
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_uri(uri: Union[str, Path]) -> bool:
    """Check if uri is valid."""
    try:
        result = urlparse(str(uri))
        return all([result.scheme, result.netloc])
    except (ValueError, AttributeError):
        return False
