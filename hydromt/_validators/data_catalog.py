"""Pydantic models for the validation of Data catalogs."""
from logging import warning
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic import AnyUrl, BaseModel, ConfigDict, model_validator
from pydantic.fields import Field
from pydantic_core import Url
from pyproj import CRS
from pyproj.exceptions import CRSError

from hydromt import __version__ as HYDROMT_VERSION
from hydromt._typing import Bbox, Number, TimeRange
from hydromt.data_catalog import _yml_from_uri_or_path


class SourceSpecDict(BaseModel):
    """A complete source variant specification."""

    source: str
    provider: Optional[str] = None
    version: Optional[Union[str, int]] = None

    @staticmethod
    def from_dict(input_dict):
        """Create a source variant specification from a dictionary."""
        return SourceSpecDict(**input_dict)


class SourceVariant(BaseModel):
    """A variant for a data source."""

    provider: Optional[Literal["local", "aws", "gcs"]] = None
    version: Optional[Union[str, Number]] = None
    path: Path
    rename: Optional[Dict[str, str]] = None
    filesystem: Optional[Literal["local", "s3", "gcs"]] = None
    storage_options: Optional[Dict[str, Any]] = None


class Extent(BaseModel):
    """A validation model for describing the space and time a dataset covers."""

    time_range: TimeRange
    bbox: Bbox


class DataCatalogMetaData(BaseModel):
    """The metadata section of a Hydromt data catalog."""

    roots: Optional[List[Path]] = None
    version: Optional[Union[str, Number]] = None
    hydromt_version: Optional[str] = None
    name: Optional[str] = None
    model_config: ConfigDict = ConfigDict(
        str_strip_whitespace=True,
        extra="allow",
    )

    @model_validator(mode="after")
    def _check_version_compatible(self) -> "DataCatalogMetaData":
        if self.hydromt_version is None:
            warning(
                f"No hydromt version was specified for the data catalog, thus compatability between used hydromt version ({HYDROMT_VERSION}) and the catalog could not be determined."
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
    def from_dict(input_dict: Dict) -> "DataCatalogMetaData":
        """Convert a dictionary into a validated data catalog metadata item."""
        return DataCatalogMetaData(**input_dict)


class DataCatalogItemMetadata(BaseModel):
    """The metadata for a data source."""

    category: Optional[str] = None
    paper_doi: Optional[str] = None
    paper_ref: Optional[str] = None
    source_license: Optional[str] = None
    source_url: Optional[AnyUrl] = None
    source_version: Optional[str] = None
    notes: Optional[str] = None

    model_config: ConfigDict = ConfigDict(
        str_strip_whitespace=True, coerce_numbers_to_str=True
    )

    @staticmethod
    def from_dict(input_dict):
        """Convert a dictionary into a validated source metadata item."""
        if input_dict is None:
            return DataCatalogItemMetadata()
        else:
            item_source_url = input_dict.pop("source_url", None)
            if item_source_url:
                Url(item_source_url)
            return DataCatalogItemMetadata(**input_dict, source_url=item_source_url)


class DataCatalogItem(BaseModel):
    """A validated data source."""

    name: str
    data_type: Literal["RasterDataset", "GeoDataset", "GeoDataFrame", "DataFrame"]
    driver: Literal[
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
    path: Optional[Path] = None
    crs: Optional[Union[int, str]] = None
    filesystem: Optional[str] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    storage_options: Dict[str, Any] = Field(default_factory=dict)
    placeholders: Optional[Dict[str, Any]] = None
    rename: Dict[str, str] = Field(default_factory=dict)
    nodata: Optional[Number] = None
    meta: Optional[DataCatalogItemMetadata] = None
    unit_add: Optional[Dict[str, Number]] = None
    unit_mult: Optional[Dict[str, Number]] = None
    variants: Optional[List[SourceVariant]] = None
    version: Optional[Union[str, Number]] = None

    model_config: ConfigDict = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    @model_validator(mode="after")
    def _check_valid_crs(self) -> "DataCatalogItem":
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
        item_metadata = DataCatalogItemMetadata.from_dict(input_dict.pop("meta", {}))
        item_kwargs = input_dict.pop("kwargs", {})
        item_storage_options = input_dict.pop("storage_options", {})
        return DataCatalogItem(
            **input_dict,
            name=entry_name,
            kwargs=item_kwargs,
            storage_options=item_storage_options,
            meta=item_metadata,
        )


class DataCatalogValidator(BaseModel):
    """A validated complete data catalog."""

    meta: Optional[DataCatalogMetaData] = None
    sources: Dict[str, DataCatalogItem] = Field(default_factory=dict)
    aliases: Dict[str, str] = Field(default_factory=dict)

    model_config: ConfigDict = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    @staticmethod
    def from_dict(input_dict):
        """Create a validated datacatalog from a dictionary."""
        if input_dict is None:
            return DataCatalogValidator()
        else:
            meta = input_dict.pop("meta", None)
            catalog_meta = DataCatalogMetaData.from_dict(meta)
            catalog_entries = {}
            catalog_aliases = {}
            for entry_name, entry_dict in input_dict.items():
                if "alias" in entry_dict.keys():
                    catalog_aliases[entry_name] = entry_dict["alias"]
                else:
                    catalog_entries[entry_name] = DataCatalogItem.from_dict(
                        entry_dict, name=entry_name
                    )

            for src, dst in catalog_aliases.items():
                assert (
                    dst in catalog_entries.keys()
                ), f"{src} references unfound entry {dst}"
            return DataCatalogValidator(
                meta=catalog_meta, sources=catalog_entries, aliases=catalog_aliases
            )

    @staticmethod
    def from_yml(path: str):
        """Create a validated datacatalog loaded from the provided path."""
        yml_dict = _yml_from_uri_or_path(path)
        return DataCatalogValidator.from_dict(yml_dict)
