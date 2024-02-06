"""Abstract DataSource class."""
from os.path import abspath, join
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from hydromt.data_adapter.caching import _uri_validator
from hydromt.metadata_resolvers import MetaDataResolver
from hydromt.metadata_resolvers.resolver_plugin import RESOLVERS
from hydromt.typing import DataType


class DataSource(BaseModel):
    """
    A DataSource is a parsed section of a DataCatalog.

    The DataSource, specific for a data type within HydroMT, is responsible for
    validating the input from the DataCatalog, to
    ensure the workflow fails as early as possible. A DataSource has information on
    the driver that the data should be read with, and is responsible for initializing
    this driver.
    """

    @classmethod
    def submodel_validate(
        cls,
        obj: Any,
        *,
        strict: Optional[bool] = None,
        from_attributes: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> "DataSource":
        """Validate the DataSource."""
        if data_type := obj.get("data_type"):
            try:
                target_cls: DataSource = next(
                    filter(lambda sc: sc.data_type == data_type, cls.__subclasses__())
                )  # subclasses should be loaded from __init__.py
                return target_cls.model_validate(
                    obj,
                    strict=strict,
                    from_attributes=from_attributes,
                    context=context,
                )
            except StopIteration:
                raise ValueError(f"Unknown 'data_type': '{data_type}'")

        raise ValueError("DataSource needs 'data_type'")

    @field_validator("metadata_resolver", mode="before")
    @classmethod
    def _validate_metadata_resolver(cls, v: Any):
        if isinstance(v, str):
            if v not in RESOLVERS:
                raise ValueError(f"unknown MetaDataResolver: '{v}'.")
            return RESOLVERS.get(v)()
        elif isinstance(v, MetaDataResolver):
            return v
        else:
            raise ValueError("metadata_resolver should be string or MetaDataResolver.")

    @field_validator("uri", mode="after")
    @classmethod
    def _validate_uri(cls, v: str, info: ValidationInfo) -> str:
        if not _uri_validator(v):
            return _abs_path(info.data.get("root"), v)

    data_type: ClassVar[DataType]
    root: Optional[str] = Field(default=None)
    name: str
    driver: Any
    attrs: Dict[str, Any] = Field(default_factory=dict)
    version: Optional[str] = Field(default=None)
    provider: Optional[str] = Field(default=None)
    metadata_resolver: MetaDataResolver
    driver_kwargs: Dict[str, Any] = Field(default_factory=dict)
    unit_add: Dict[str, Any] = Field(default_factory=dict)
    unit_mult: Dict[str, Any] = Field(default_factory=dict)
    rename: Dict[str, str] = Field(default_factory=dict)
    nodata: Union[float, int, Dict[str, Union[float, int]]] = Field(default=None)
    extent: Dict[str, Any] = Field(default_factory=dict)  # ?
    meta: Dict[str, Any] = Field(default_factory=dict)
    uri: str
    crs: Optional[int] = Field(default=None)


def _abs_path(root: Union[Path, str], rel_path: Union[Path, str]) -> str:
    path = Path(str(rel_path))
    if not path.is_absolute():
        if root is not None:
            rel_path = join(root, rel_path)
        path = Path(abspath(rel_path))
    return str(path)
