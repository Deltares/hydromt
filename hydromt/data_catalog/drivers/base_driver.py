"""Base class for different drivers."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hydromt._abstract_base import AbstractBaseModel
from hydromt.error import NoDataStrategy
from hydromt.plugins import PLUGINS
from hydromt.typing.fsspec_types import (
    FSSpecFileSystem,
)

logger = logging.getLogger(__name__)

DRIVER_OPTIONS_DESCRIPTION = """
Driver options that can be used to configure the behavior of the driver.
DriverOptions allows for setting arbitrary kwargs. Any options not explicitly declared
in the DriverOptions class are passed as kwargs to the underlying `open` functions.
"""


class DriverOptions(BaseModel):
    """Options for the driver.

    Fields declared in this class or its subclasses are used to configure the behavior of the driver.
    Any other undeclared keyword arguments are allowed and will be stored in the instance.
    These extra keyword arguments are passed as kwargs to the underlying `open` functions used by the driver.

    Retrieving the extra kwargs can be done using the `get_kwargs` method:
    - Some options may be solely used internally by the driver and not passed to the `open` functions.
    - Some options may be used both internally and passed to the `open` functions.
    - To specify which declared fields should be passed to the `open` functions, set the class variable `KWARGS_FOR_OPEN`.
    - By default, all declared fields are excluded, and only extra kwargs are passed.

    """

    # allow arbitrary kwargs, usually passed to some `open_dataset` function()
    model_config = ConfigDict(extra="allow")

    KWARGS_FOR_OPEN: ClassVar[set[str]] = set()
    """Fields that are to be included as kwargs when passing to `open` functions.
    By default, all fields defined in the class are excluded."""

    @classmethod
    def get_excluded_kwarg_names_for_open(cls) -> set[str]:
        """Get the fields that are to be excluded when passing to `open` functions.

        This includes all declared fields in the class, minus those defined in `KWARGS_FOR_OPEN`.
        """
        return set(cls.model_fields.keys()).difference(cls.KWARGS_FOR_OPEN)

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
    """"Whether the driver supports writing data."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    """"Forbid extra fields in the driver configuration and allow arbitrary types."""

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = set()
    """"Set of supported file extensions for this driver."""

    filesystem: FSSpecFileSystem = Field(
        default_factory=FSSpecFileSystem,
        description="Filesystem to use for accessing the data.",
    )
    options: DriverOptions = Field(
        default_factory=DriverOptions, description=DRIVER_OPTIONS_DESCRIPTION
    )

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

    @abstractmethod
    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        kwargs_for_open: dict[str, Any] | None = None,
    ):
        """Read data using the driver.

        Parameters
        ----------
        uris : list[str]
            List of URIs to read data from.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle no data situations. Default is NoDataStrategy.RAISE.
        kwargs_for_open : dict[str, Any] | None, optional
            Additional keyword arguments to pass to the underlying open function. Default is None.
        """
        ...

    @abstractmethod
    def write(self, path: str | Path, *args, **kwargs):
        """Write data using the driver.

        Parameters
        ----------
        path : str | Path
            Path to write data to.
        *args
            Positional arguments for the write function.
        **kwargs
            Keyword arguments for the write function.
        """
        ...

    @field_validator("filesystem", mode="before")
    @classmethod
    def _create_filesystem(cls, v: Any) -> FSSpecFileSystem:
        """Allow filesystem to be provided as string, dict, or FSSpecFileSystem."""
        return FSSpecFileSystem.create(v)
