"""Base class for different drivers."""

from abc import ABC
from logging import Logger, getLogger
from typing import Any, ClassVar, Dict

from fsspec.implementations.local import LocalFileSystem
from pydantic import (
    ConfigDict,
    Field,
)

from hydromt._abstract_base import AbstractBaseModel
from hydromt._typing import FS
from hydromt.plugins import PLUGINS

logger: Logger = getLogger(__name__)


class BaseDriver(AbstractBaseModel, ABC):
    """Base class for different drivers.

    Is used to implement common functionality.
    """

    supports_writing: ClassVar[bool] = False
    filesystem: FS = Field(default_factory=LocalFileSystem)
    options: Dict[str, Any] = Field(default_factory=dict)

    model_config: ConfigDict = ConfigDict(extra="forbid")

    def __eq__(self, value: object) -> bool:
        """Compare equality.

        Overwritten as filesystem between two sources can not be the same.
        """
        if isinstance(value, self.__class__):
            return self.model_dump(round_trip=True) == value.model_dump(round_trip=True)
        return super().__eq__(value)

    @classmethod
    def _load_plugins(cls):
        """Load Driver plugins."""
        plugins: Dict[str, BaseDriver] = PLUGINS.uri_resolver_plugins
        logger.debug(f"loaded {cls.__name__} plugins: {plugins}")
