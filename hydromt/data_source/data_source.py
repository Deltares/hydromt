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
from hydromt.data_adapter.data_adapter_base import DataAdapterBase
from hydromt.data_adapter.harmonization_settings import HarmonizationSettings
from hydromt.driver import BaseDriver


class DataSource(BaseModel):
    """
    A DataSource is a parsed section of a DataCatalog.

    The DataSource, specific for a data type within HydroMT, is responsible for
    validating the input from the DataCatalog, to
    ensure the workflow fails as early as possible. A DataSource has information on
    the driver that the data should be read with, and is responsible for initializing
    this driver.
    """

    name: str
    uri: str
    data_adapter: DataAdapterBase
    driver: BaseDriver
    data_type: ClassVar[DataType]
    root: Optional[str] = Field(default=None)
    harmonization: HarmonizationSettings = Field(default_factory=HarmonizationSettings)
    version: Optional[str] = Field(default=None)
    provider: Optional[str] = Field(default=None)
    driver_kwargs: Dict[str, Any] = Field(default_factory=dict)
    resolver_kwargs: Dict[str, Any] = Field(default_factory=dict)
    crs: Optional[int] = Field(default=None)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the DataSource."""
        summ: Dict[str, Any] = self.model_dump(include={"uri"})
        summ.update(
            {
                "data_type": self.__class__.data_type,
                "driver": self.driver.__repr_name__(),
                **self.data_adapter.harmonization_settings.meta,
            }
        )
        return summ

    # TODO: def to_file(self, **query_params) https://github.com/Deltares/hydromt/issues/840

    @model_validator(mode="wrap")
    @classmethod
    def _init_subclass_data_type(cls, data: Any, handler: Callable):
        """Initialize the subclass based on the 'data_type' class variable.

        All DataSources should be parsed based on their `data_type` class variable;
        e.g. a dict with `data_type` = `RasterDataset` should be parsed as a
        `RasterDatasetSource`. This class searches all subclasses until the correct
        `DataSource` is found and initialized that one.
        This allow an API as: `DataSource.model_validate(raster_ds_dict)` or
        `DataSource(raster_ds_dict)`, but also `RasterDatasetSource(raster_ds_dict)`.

        Inspired by: https://github.com/pydantic/pydantic/discussions/7008#discussioncomment
        This validator does not yet support submodels of submodels.
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

        raise ValueError(f"{cls.__name__} needs 'data_type'")

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


def _abs_path(root: Union[Path, str], rel_path: Union[Path, str]) -> str:
    path = Path(str(rel_path))
    if not path.is_absolute():
        if root is not None:
            rel_path = join(root, rel_path)
        path = Path(abspath(rel_path))
    return str(path)
