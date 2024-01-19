"""Abstract Driver class."""
from typing import Any, Mapping


class BaseDriver:
    """Base class for Driver."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Getter for uri."""
        return self._kwargs
