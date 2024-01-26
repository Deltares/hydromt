"""Abstract DataSource class."""
from typing import Any

from pydantic import BaseModel, Field, field_validator

from hydromt import DataCatalog
from hydromt.metadata_resolvers import MetaDataResolver
from hydromt.metadata_resolvers.resolver_plugin import RESOLVERS


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
    def from_catalog(
        cls,
        catalog: DataCatalog,
        key: str,
        provider: str | None = None,
        version: str | None = None,
    ) -> "DataSource":
        """Create Data source from DataCatalog."""
        return cls.model_validate(
            catalog.get_source(key, provider, version).update("name", key)
        )

    @field_validator("metadata_resolver", mode="before")
    def _validate_metadata_resolver(cls, v: Any):
        if isinstance(v, str):
            assert v in RESOLVERS, f"unknown MetaDataResolver: '{v}'."
            return RESOLVERS.get(v)()
        elif isinstance(v, MetaDataResolver):
            return v
        else:
            raise ValueError("metadata_resolver should be string or MetaDataResolver.")

    name: str
    data_type: str
    attrs: dict[str, Any] = Field(default_factory=dict)
    version: str | None = Field(default=None)
    provider: str | None = Field(default=None)
    metadata_resolver: MetaDataResolver
    driver_kwargs: dict[str, Any] = Field(default_factory=dict)
    unit_add: dict[str, Any] = Field(default_factory=dict)
    unit_mult: dict[str, Any] = Field(default_factory=dict)
    rename: dict[str, str] = Field(default_factory=dict)
    nodata: float | int | dict[str, float | int] | None = Field(default=None)
    extent: dict[str, Any] = Field(default_factory=dict)  # ?
    meta: dict[str, Any] = Field(default_factory=dict)
    uri: str
    crs: int | None = Field(default=None)
