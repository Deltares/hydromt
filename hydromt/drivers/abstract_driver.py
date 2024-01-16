"""Abstract Driver class."""
from abc import ABC


class AbstractDriver(ABC):
    """Abstract class for Driver to implement."""

    def __init__(self, uri: str, **kwargs):
        self._uri = uri

    _uri: str

    @property
    def uri(self) -> str:
        """Getter for uri."""
        return self._uri
