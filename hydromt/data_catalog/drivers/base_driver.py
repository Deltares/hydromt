"""Base class for different drivers."""

import logging
from abc import ABC
from typing import Any, ClassVar, Dict

from fsspec.implementations.local import LocalFileSystem
from pydantic import (
    ConfigDict,
    Field,
)

from hydromt._abstract_base import AbstractBaseModel
from hydromt._typing import FS
from hydromt.plugins import PLUGINS

logger = logging.getLogger(__name__)


class BaseDriver(AbstractBaseModel, ABC):
    """Base class for different drivers.

    Is used to implement common functionality.
    """

    supports_writing: ClassVar[bool] = False
    filesystem: FS = Field(default_factory=LocalFileSystem)
    options: Dict[str, Any] = Field(default_factory=dict)

    model_config: ConfigDict = ConfigDict(extra="forbid")
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = set()

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
        plugins = PLUGINS.driver_plugins.keys()
        logger.debug(f"loaded {cls.__name__} plugins: {', '.join(plugins)}")
