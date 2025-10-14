"""Pydantic models for the validation of Data catalogs."""

from logging import warning
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    model_serializer,
    model_validator,
)
from pydantic.fields import Field
from pydantic_core import Url
from pyproj import CRS
from pyproj.exceptions import CRSError

from hydromt import __version__ as HYDROMT_VERSION
from hydromt.io.readers import _yml_from_uri_or_path
from hydromt.typing import Bbox, Number, TimeRange

DEFAULT_DRIVER_MAPPING = {
    "RasterDataset": "raster",
    "GeoDataFrame": "vector",
    "DataFrame": "csv",
    "GeoDataset": "vector",
}


class SourceSpecDict(BaseModel):
    """A complete source variant specification."""

    source: str
    provider: Optional[str] = None
    version: Optional[Union[str, Number]] = None

    @staticmethod
    def from_dict(input_dict):
        """Create a source variant specification from a dictionary."""
        return SourceSpecDict(**input_dict)


class SourceVariant(BaseModel):
    """A variant for a data source."""

    provider: str | None = None
    version: Optional[Union[str, Number]] = None
    path: Path
    rename: Optional[Dict[str, str]] = None
    filesystem: Optional[Literal["local", "s3", "gcs"]] = None
    storage_options: Optional[Dict[str, Any]] = None


class Extent(BaseModel):
    """A validation model for describing the space and time a dataset covers."""

    time_range: TimeRange
    bbox: Bbox


class DataCatalogV0MetaData(BaseModel):
    """The metadata section of a Hydromt data catalog."""

    roots: Optional[List[Path]] = None
    version: Optional[Union[str, Number]] = None
    hydromt_version: Optional[str] = None
    name: Optional[str] = None
    validate_hydromt_version: bool = True
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="allow",
    )

    @model_validator(mode="after")
    def _check_version_compatible(self) -> "DataCatalogV0MetaData":
        if not self.validate_hydromt_version:
            return self

        if self.hydromt_version is None:
            warning(
                f"No hydromt version was specified for the data catalog, thus compatibility between used hydromt version ({HYDROMT_VERSION}) and the catalog could not be determined."
            )
            return self
        requested = SpecifierSet(self.hydromt_version, prereleases=True)
        version = Version(HYDROMT_VERSION)

        if version in requested:
            return self
        else:
            raise ValueError(
                f"Current hydromt version {HYDROMT_VERSION} is not compatible with version specified in catalog {self.hydromt_version}"
            )

    @staticmethod
    def from_dict(input_dict: Dict[str, Any]) -> "DataCatalogV0MetaData":
        """Convert a dictionary into a validated data catalog metadata item."""
        return DataCatalogV0MetaData(**input_dict)


class DataCatalogV0ItemMetadata(BaseModel):
    """The metadata for a data source."""

    category: Optional[str] = None
    paper_doi: Optional[str] = None
    paper_ref: Optional[str] = None
    source_license: Optional[str] = None
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    notes: Optional[str] = None
    temporal_extent: Optional[dict] = None
    spatial_extent: Optional[dict] = None

    model_config = ConfigDict(str_strip_whitespace=True, coerce_numbers_to_str=True)

    @model_serializer
    def serialize(self):
        if self.is_empty():
            return None
        else:
            return self.model_dump(
                exclude_defaults=True, exclude_none=True, exclude_unset=True
            )

    @staticmethod
    def from_dict(input_dict):
        """Convert a dictionary into a validated source metadata item."""
        if input_dict is None:
            return DataCatalogV0ItemMetadata()
        else:
            item_source_url = input_dict.pop("source_url", None)
            if item_source_url:
                Url(item_source_url)
            return DataCatalogV0ItemMetadata(**input_dict, source_url=item_source_url)

    def is_empty(self):
        return not any(
            attr is not None
            for attr in [
                self.category,
                self.paper_doi,
                self.paper_ref,
                self.source_license,
                self.source_url,
                self.source_version,
                self.notes,
                self.temporal_extent,
                self.spatial_extent,
            ]
        )


class DataCatalogV0Item(BaseModel):
    """A validated data source."""

    name: str
    data_type: Literal["RasterDataset", "GeoDataset", "GeoDataFrame", "DataFrame"]
    driver: (
        Literal[
            "csv",
            "fwf",
            "netcdf",
            "parquet",
            "raster",
            "raster_tindex",
            "vector",
            "vector_table",
            "xls",
            "xlsx",
            "zarr",
        ]
        | None
    ) = None
    path: Optional[Path] = None
    crs: Optional[Union[int, str]] = None
    filesystem: Optional[str] = None
    provider: Optional[str] = None
    driver_kwargs: Dict[str, Any] = Field(default_factory=dict)
    kwargs: Dict[str, Any] = Field(default_factory=dict)  # deprecated
    storage_options: Dict[str, Any] = Field(default_factory=dict)
    placeholders: Optional[Dict[str, Any]] = None
    rename: Dict[str, str] = Field(default_factory=dict)
    nodata: Optional[Number] = None
    meta: Optional[DataCatalogV0ItemMetadata] = None
    unit_add: Optional[Dict[str, Number]] = None
    unit_mult: Optional[Dict[str, Number]] = None
    variants: Optional[List[SourceVariant]] = None
    version: Optional[Union[str, Number]] = None

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    @model_validator(mode="after")
    def _check_valid_crs(self) -> "DataCatalogV0Item":
        try:
            if self.crs:
                _ = CRS.from_user_input(self.crs)
        except CRSError as e:
            raise ValueError(e)
        return self

    @staticmethod
    def from_dict(input_dict, name=None):
        """Convert a dictionary into a validated source item."""
        dict_name = input_dict.pop("name", None)
        if name is None:
            entry_name = dict_name
        else:
            entry_name = name
        try:
            item_metadata = DataCatalogV0ItemMetadata.from_dict(
                input_dict.pop("meta", {})
            )
            item_kwargs = input_dict.pop("kwargs", {})

            item_storage_options = input_dict.pop("storage_options", {})

            return DataCatalogV0Item(
                **input_dict,
                name=entry_name,
                kwargs=item_kwargs,
                storage_options=item_storage_options,
                meta=item_metadata,
            )
        except ValidationError as e:
            raise ValidationError.from_exception_data(
                entry_name or "nameless entry", e.errors(), "python"
            )


class DataCatalogV0Validator(BaseModel):
    """A validated complete data catalog."""

    meta: DataCatalogV0MetaData | None = None
    sources: Dict[str, DataCatalogV0Item] = Field(default_factory=dict)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    @staticmethod
    def from_dict(input_dict):
        """Create a validated datacatalog from a dictionary."""
        if input_dict is None:
            return DataCatalogV0Validator()
        else:
            if meta := input_dict.pop("meta", None):
                catalog_meta = DataCatalogV0MetaData.from_dict(meta)
            else:
                catalog_meta = None

            catalog_entries = {}
            for entry_name, entry_dict in input_dict.items():
                catalog_entries[entry_name] = DataCatalogV0Item.from_dict(
                    entry_dict, name=entry_name
                )

            return DataCatalogV0Validator(meta=catalog_meta, sources=catalog_entries)

    @staticmethod
    def from_yml(path: str):
        """Create a validated data catalog loaded from the provided path."""
        yml_dict = _yml_from_uri_or_path(path)
        return DataCatalogV0Validator.from_dict(yml_dict)
