"""Base class for different drivers."""

from abc import ABC
from logging import Logger, getLogger
from typing import ClassVar

from fsspec.implementations.local import LocalFileSystem
from pydantic import BaseModel, ConfigDict, Field

from hydromt._abstract_base import AbstractBaseModel
from hydromt._typing import FS
from hydromt.plugins import PLUGINS

logger: Logger = getLogger(__name__)


class DriverOptions(BaseModel):
    """Options for the driver."""

    # allow arbitrary kwargs, usually passed to some `open_dataset` function()
    model_config = ConfigDict(extra="allow")

    def to_dict(self, exclude: set[str] | None = None) -> dict:
        """Return dict of kwargs excluding reserved/internal keys."""
        return self.model_dump(exclude=exclude, exclude_unset=True)


class BaseDriver(AbstractBaseModel, ABC):
    """Base class for different drivers.

    Is used to implement common functionality.
    """

    supports_writing: ClassVar[bool] = False
    filesystem: FS = Field(default_factory=LocalFileSystem)
    options: DriverOptions = Field(default_factory=DriverOptions)

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
