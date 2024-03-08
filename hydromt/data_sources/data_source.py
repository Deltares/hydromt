"""Abstract DataSource class."""
from os.path import abspath, join
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

from hydromt._typing import DataType
from hydromt.data_adapter.caching import _uri_validator
from hydromt.data_adapter.harmonization_settings import HarmonizationSettings
from hydromt.drivers import BaseDriver


class DataSource(BaseModel):
    """
    A DataSource is a parsed section of a DataCatalog.

    The DataSource, specific for a data type within HydroMT, is responsible for
    validating the input from the DataCatalog, to
    ensure the workflow fails as early as possible. A DataSource has information on
    the driver that the data should be read with, and is responsible for initializing
    this driver.
    """

    @model_validator(mode="wrap")
    @classmethod
    def _init_subclass_data_type(cls, data: Any, handler: Callable):
        """Initialize the subclass based on the 'data_type' class variable.

        this does not yet support submodels of submodels. If we need more hierarchy, check:
        https://github.com/pydantic/pydantic/discussions/7008#discussioncomment-7966076.
        """
        if not isinstance(data, dict):
            # Other objects should already be the correct subclass.
            return handler(data)

        if DataSource in cls.__bases__:
            # If cls is subclass DataSource, just validate as normal
            return handler(data)

        if data_type := data.get("data_type"):
            try:
                # Find which DataSource to instantiate.
                target_cls: DataSource = next(
                    filter(lambda sc: sc.data_type == data_type, cls.__subclasses__())
                )  # subclasses should be loaded from __init__.py
                return target_cls.model_validate(data)
            except StopIteration:
                raise ValueError(f"Unknown 'data_type': '{data_type}'")

        raise ValueError("DataSource needs 'data_type'")

    @model_validator(mode="before")
    @classmethod
    def _validate_data_type(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and data.get("data_type")
            and data.get("data_type") != cls.data_type
        ):
            raise ValueError(f"'data_type' must be '{cls.data_type}'.")
        return data

    @model_validator(mode="after")
    def _validate_uri(self) -> str:
        if not _uri_validator(self.uri):
            self.uri = _abs_path(self.root, self.uri)
        return self

    root: Optional[str] = Field(default=None)
    name: str
    uri: str
    data_type: ClassVar[DataType]
    driver: BaseDriver
    harmonization: HarmonizationSettings = Field(default_factory=HarmonizationSettings)
    # data_adapter: Any
    version: Optional[str] = Field(default=None)
    provider: Optional[str] = Field(default=None)
    driver_kwargs: Dict[str, Any] = Field(default_factory=dict)
    resolver_kwargs: Dict[str, Any] = Field(default_factory=dict)
    crs: Optional[int] = Field(default=None)


def _abs_path(root: Union[Path, str], rel_path: Union[Path, str]) -> str:
    path = Path(str(rel_path))
    if not path.is_absolute():
        if root is not None:
            rel_path = join(root, rel_path)
        path = Path(abspath(rel_path))
    return str(path)
