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
    """Options for the driver.

    Fields declared in this class or its subclasses are used to configure the behavior of the driver.
    Any other undeclared keyword arguments are allowed and will be stored in the instance.
    These extra keyword arguments are passed as kwargs to the underlying `open` functions used by the driver.

    Retrieving the extra kwargs can be done using the `get_kwargs` method:
    - Some options may be solely used internally by the driver and not passed to the `open` functions.
    - Some options may be used both internally and passed to the `open` functions.
    - To specify which declared fields should be passed to the `open` functions, set the class variable `_kwargs_for_open`.
    - By default, all declared fields are excluded, and only extra kwargs are passed.

    """

    # allow arbitrary kwargs, usually passed to some `open_dataset` function()
    model_config = ConfigDict(extra="allow")

    _kwargs_for_open: ClassVar[set[str]] = set()
    """Fields that are to be included as kwargs when passing to `open` functions.
    By default, all fields defined in the class are excluded."""

    @classmethod
    def get_excluded_kwarg_names_for_open(cls) -> set[str]:
        """Get the fields that are to be excluded when passing to `open` functions.

        This includes all declared fields in the class, minus those defined in `_kwargs_for_open`.
        """
        return set(cls.model_fields.keys()).difference(cls._kwargs_for_open)

    def to_dict(self, exclude: set[str] | None = None) -> dict:
        """Return dict of kwargs excluding reserved/internal keys, and including the extra kwargs."""
        return self.model_dump(exclude=exclude, exclude_unset=True)

    def get_kwargs(self) -> dict:
        """Return attributes set that are not explicitly declared fields."""
        return self.model_dump(
            exclude=self.get_excluded_kwarg_names_for_open(), exclude_unset=True
        )


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
