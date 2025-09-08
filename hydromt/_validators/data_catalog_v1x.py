"""Pydantic models for the validation of Data catalogs."""

from logging import warning
from pathlib import Path
from typing import Any, Dict, List, Literal

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic import AnyUrl, BaseModel, ConfigDict, field_validator, model_validator
from pydantic.fields import Field
from pydantic_core import Url, ValidationError
from pyproj import CRS
from pyproj.exceptions import CRSError

from hydromt import __version__ as HYDROMT_VERSION
from hydromt._io.readers import _yml_from_uri_or_path
from hydromt._typing import Bbox, Number, TimeRange


class SourceSpecDict(BaseModel):
    """A complete source variant specification."""

    source: str
    provider: str | None = None
    version: str | Number | None = None

    @staticmethod
    def from_dict(input_dict):
        """Create a source variant specification from a dictionary."""
        return SourceSpecDict(**input_dict)


class DataCatalogV1UriResolverItem(BaseModel):
    name: str
    options: Dict[str, Any] | None = None


class DataCatalogV1DriverItem(BaseModel):
    name: str
    options: Dict[str, Any] | None = None


class SourceVariant(BaseModel):
    """A variant for a data source."""

    provider: str | None = None
    version: str | Number | None = None
    uri: Path | None = None
    rename: Dict[str, str] | None = None
    filesystem: Literal["local", "s3", "gcs"] | None = None
    storage_options: Dict[str, Any] | None = None
    driver: DataCatalogV1DriverItem | None = None


class Extent(BaseModel):
    """A validation model for describing the space and time a dataset covers."""

    time_range: TimeRange
    bbox: Bbox


class DataCatalogV1MetaData(BaseModel):
    """The metadata section of a Hydromt data catalog."""

    roots: List[Path] | None = None
    version: str | Number | None = None
    hydromt_version: str | None = None
    name: str | None = None
    category: str | None = None
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="allow",
    )

    @model_validator(mode="after")
    def _check_version_compatible(self) -> "DataCatalogV1MetaData":
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
    def from_dict(input_dict: Dict[str, Any]) -> "DataCatalogV1MetaData":
        """Convert a dictionary into a validated data catalog metadata item."""
        return DataCatalogV1MetaData(**input_dict)


class DataCatalogV1ItemMetadata(BaseModel):
    """The metadata for a data source."""

    crs: str | int | None = None
    category: str | None = None
    paper_doi: str | None = None
    paper_ref: str | None = None
    source_license: str | None = None
    source_url: AnyUrl | None = None
    source_version: str | None = None
    notes: str | None = None
    temporal_extent: dict | None = None
    spatial_extent: dict | None = None

    model_config = ConfigDict(str_strip_whitespace=True, coerce_numbers_to_str=True)

    @field_validator("crs")
    def _check_valid_crs(v):
        try:
            _ = CRS.from_user_input(v)
        except CRSError as e:
            raise ValueError(e)
        return v

    @staticmethod
    def from_dict(input_dict):
        """Convert a dictionary into a validated source metadata item."""
        if input_dict is None:
            return DataCatalogV1ItemMetadata()
        else:
            item_source_url = input_dict.pop("source_url", None)
            if item_source_url:
                Url(item_source_url)
            return DataCatalogV1ItemMetadata(**input_dict, source_url=item_source_url)


class DataCatalogV1DataAdapter(BaseModel):
    rename: Dict[str, str] | None = None
    unit_mult: Dict[str, int | float] | None = None
    unit_add: Dict[str, int | float] | None = None


class DataCatalogV1Item(BaseModel):
    """A validated data source."""

    # these are required but they can be defined in the variants instead
    driver: DataCatalogV1DriverItem | None = None
    uri: str | None = None

    uri_resolver: DataCatalogV1UriResolverItem | None = None
    name: str
    data_type: Literal["RasterDataset", "GeoDataset", "GeoDataFrame", "DataFrame"]
    provider: str | None = None
    metadata: DataCatalogV1ItemMetadata | None = None
    variants: List[SourceVariant] | None = None
    version: str | Number | None = None
    placehodler: dict[str, list[str]] | None = None
    data_adapter: DataCatalogV1DataAdapter | None = None

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    @model_validator(mode="after")
    def driver_in_item_or_variants(cls, values):
        if values.driver is not None or (
            values.variants is not None
            and all([variant.driver is not None for variant in values.variants])
        ):
            return values
        else:
            raise ValueError(
                "Source must either have a driver, or all of it's variants must have one."
            )

    @model_validator(mode="after")
    def uri_in_item_or_variants(cls, values):
        if values.uri is not None or (
            values.variants is not None
            and all([variant.uri is not None for variant in values.variants])
        ):
            return values
        else:
            raise ValueError(
                "Source must either have a uri, or all of it's variants must have one."
            )

    @staticmethod
    def from_dict(input_dict, name=None):
        """Convert a dictionary into a validated source item."""
        dict_name = input_dict.pop("name", None)
        if name is None:
            entry_name = dict_name
        else:
            entry_name = name
        try:
            item_metadata = DataCatalogV1ItemMetadata.from_dict(
                input_dict.pop("metadata", {})
            )

            return DataCatalogV1Item(
                **input_dict,
                name=entry_name,
                metadata=item_metadata,
            )
        except ValidationError as e:
            raise ValidationError.from_exception_data(
                entry_name or "nameless entry", e.errors(), "python"
            )


class DataCatalogV1Validator(BaseModel):
    """A validated complete data catalog."""

    meta: DataCatalogV1MetaData | None = None
    sources: Dict[str, DataCatalogV1Item] = Field(default_factory=dict)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    @staticmethod
    def from_dict(input_dict):
        """Create a validated datacatalog from a dictionary."""
        if input_dict is None:
            return DataCatalogV1Validator()
        else:
            meta = input_dict.pop("meta", None)
            catalog_meta = DataCatalogV1MetaData.from_dict(meta)
            catalog_entries = {}
            for entry_name, entry_dict in input_dict.items():
                catalog_entries[entry_name] = DataCatalogV1Item.from_dict(
                    entry_dict, name=entry_name
                )

            return DataCatalogV1Validator(meta=catalog_meta, sources=catalog_entries)

    @staticmethod
    def from_yml(path: str):
        """Create a validated data catalog loaded from the provided path."""
        yml_dict = _yml_from_uri_or_path(path)
        return DataCatalogV1Validator.from_dict(yml_dict)
