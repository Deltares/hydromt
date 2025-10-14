"""Abstract DataSource class."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from os.path import abspath, join, splitext
from pathlib import Path, PurePath
from typing import Any, ClassVar, Dict, List, Optional, TypeVar, Union, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from pyproj import CRS
from pystac import Catalog as StacCatalog

from hydromt._utils.uris import _is_valid_url
from hydromt.data_catalog.adapters.data_adapter_base import DataAdapterBase
from hydromt.data_catalog.drivers import BaseDriver
from hydromt.data_catalog.uri_resolvers import ConventionResolver, URIResolver
from hydromt.error import NoDataException, NoDataStrategy
from hydromt.typing import DataType, SourceMetadata
from hydromt.typing.type_def import TimeRange, TotalBounds

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DataSource(BaseModel, ABC):
    """
    A DataSource is a parsed section of a DataCatalog.

    The DataSource, specific for a data type within HydroMT, is responsible for
    validating the input from the DataCatalog, to
    ensure the workflow fails as early as possible. A DataSource has information on
    the driver that the data should be read with, and is responsible for initializing
    this driver.
    """

    model_config = ConfigDict(extra="forbid")

    _used: bool = PrivateAttr(default=False)
    _fallback_driver_read: ClassVar[str]
    _fallback_driver_write: ClassVar[str]

    name: str
    uri: str
    data_adapter: DataAdapterBase
    driver: BaseDriver
    uri_resolver: URIResolver = Field(default_factory=ConventionResolver)
    data_type: ClassVar[DataType]
    root: Optional[str] = Field(
        default=None, exclude=True
    )  # root is already in the catalog.
    version: Optional[Union[str, int, float]] = Field(default=None)
    provider: Optional[str] = Field(default=None)
    metadata: SourceMetadata = Field(default_factory=SourceMetadata)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the DataSource."""
        summ: Dict[str, Any] = self.model_dump(include={"uri"})
        summ.update(
            {
                "data_type": self.__class__.data_type,
                "driver": self.driver.__repr_name__(),
                **self.metadata.model_dump(exclude_unset=True),
            }
        )
        return summ

    def _mark_as_used(self):
        """Mark the data adapter as used."""
        self._used = True

    def _log_start_read_data(self):
        """Log the start of the read data process."""
        logger.info(f"Reading {self.name} {self.data_type} data from {self.full_uri}")

    @model_validator(mode="before")
    @classmethod
    def _validate_data_type(cls, data: Any) -> Any:
        """Pydantic does not check class variables, so it is checked here."""
        if isinstance(data, dict):
            copy_data: dict = deepcopy(data)
            if data_type := copy_data.pop("data_type", None):
                if data_type != cls.data_type:
                    raise ValueError(f"'data_type' must be '{cls.data_type}'.")
            if not copy_data.get("driver"):
                copy_data["driver"] = cls._infer_default_driver(copy_data.get("uri"))
        return copy_data

    @classmethod
    def _infer_default_driver(
        cls, uri: str | None = None, driver: type[BaseDriver] | None = None
    ) -> str:
        if uri is None:
            return cls._fallback_driver_read
        _, extension = splitext(uri)
        driver = driver if driver else BaseDriver
        return next(
            (
                driver.name
                for driver in driver.find_all_possible_types()
                if extension in driver.SUPPORTED_EXTENSIONS
            ),
            cls._fallback_driver_read,
        )

    @model_validator(mode="wrap")
    @classmethod
    def _wrap_driver_validation(cls, data, handler):
        driver = data.get("driver")
        if isinstance(driver, BaseDriver):
            data["driver"] = driver.model_dump()
        return handler(data)

    @model_validator(mode="after")
    def _validate_fs_equal_if_not_set(self) -> DataSource:
        """
        Validate and change the filesystems.

        They have to be equal between driver and uri resolver if they are not set.
        They can be different, but only if set explicitly.
        """
        driver_fs_set = "filesystem" in self.driver.model_fields_set
        uri_res_fs_set = "filesystem" in self.uri_resolver.model_fields_set
        if driver_fs_set ^ uri_res_fs_set:
            if driver_fs_set:
                self.uri_resolver.filesystem = self.driver.filesystem
            else:
                self.driver.filesystem = self.uri_resolver.filesystem
        return self

    @property
    def full_uri(self) -> str:
        """Join root with uri."""
        uri_is_url: bool = _is_valid_url(self.uri)
        if uri_is_url:
            # uri is fully self-describing
            return self.uri
        elif self.root and _is_valid_url(self.root):
            # use '/' to connect url parts
            return f"{self.root.rstrip('/')}/{self.uri}"
        # Local file, make absolute
        return _abs_path(self.root, self.uri)

    @model_serializer(mode="wrap")
    def _serialize(self, nxt: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        """Serialize data_type."""
        res: Dict[str, Any] = nxt(self)
        res["data_type"] = self.data_type

        return res

    def _get_uri_basename(self, handle_nodata: NoDataStrategy, **query_kwargs) -> str:
        if "{" in self.uri:
            # first resolve any placeholders
            uris: List[str] = self.uri_resolver.resolve(
                uri=self.full_uri,
                handle_nodata=handle_nodata,
                **query_kwargs,
            )

            # if multiple_uris, use the first one:
            if len(uris) > 0:
                uri: str = uris[0]
            else:
                raise NoDataException("!")
        else:
            uri: str = self.uri

        basename: Optional[str] = PurePath(uri).name
        if basename is None:
            raise ValueError(f"Failed to get basename of uri: {self.uri}")
        else:
            return basename

    def get_time_range(
        self, detect: bool = True, strict: bool = False
    ) -> TimeRange | None:
        """Detect the time range of the dataset if applicable.

        Override in subclasses if applicable.

        Parameters
        ----------
        detect: bool, Optional
            If True and the time range is not set in metadata, attempt to detect it.
            If False, only use the time range in metadata if it exists.

        Returns
        -------
        range: TimeRange, optional
            Instance containing the start and end of the time dimension. Range is
            inclusive on both sides. None if not set and detect is False.
        """
        time_range = self.metadata.extent.get("time_range", None)
        if time_range is not None:
            time_range = TimeRange.create(time_range)
        else:
            if detect:
                time_range = self._detect_time_range(strict=strict)

        return time_range

    def get_bbox(
        self, crs: CRS | None = None, detect: bool = True, strict: bool = False
    ) -> TotalBounds | None:
        """Get the bounding box and crs of the data source if applicable.

        This method should be overridden in subclasses if applicable.

        Returns
        -------
        bbox: TotalBounds | None
            The bounding box of the data source.
            None if not applicable.

        Notes
        -----
        TotalBounds is a tuple of (bbox, crs), where bbox is a tuple of
        (minx, miny, maxx, maxy) and crs is the coordinate reference system.
        """
        bbox = self.metadata.extent.get("bbox", None)
        crs = cast(int, crs)
        if bbox is None and detect:
            res = self._detect_bbox(strict=strict)
            if res is None:
                return None
            else:
                bbox, crs = res

        return bbox, crs

    ## Abstract methods
    @abstractmethod
    def read_data(self, **kwargs):
        """Read data from the source."""
        pass

    @abstractmethod
    def to_stac_catalog(
        self,
        handle_nodata: NoDataStrategy = NoDataStrategy.IGNORE,
    ) -> Optional[StacCatalog]:
        """
        Convert source into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.

        Parameters
        ----------
        handle_nodata: NoDataStrategy, optional
            The error handling strategy. Options are: "raise" to raise an error on
            failure, "skip" to skip the source on failure, and "coerce" (default) to
            set default values on failure.

        Returns
        -------
        StacCatalog, optional
            The STAC Catalog representation of the source, or None if the dataset was
            skipped.
        """
        ...

    @abstractmethod
    def to_file(
        self,
        file_path: Path | str,
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        driver_override: Optional[BaseDriver] = None,
        **kwargs,
    ) -> "DataSource":
        """
        Write the DataSource to a local file.

        Parameters
        ----------
        file_path: Path | str
            The path to write the data to.
        handle_nodata: NoDataStrategy, optional
            The error handling strategy. Options are: "raise" to raise an error on
            failure, "ignore" to skip the dataset on failure.
        driver_override: BaseDriver, optional
            If provided, use this driver to write the data instead of the one
            specified in the source. The driver must support writing.
        **kwargs: Any
            Additional keyword arguments for the implementation of `to_file`
            from subclasses of DataSource. Will be passed to the driver's write method.

        Returns
        -------
        DataSource
            A new instance of the DataSource with the updated uri and driver.
        """
        ...

    ## Optional overrides if applicable in subclasses
    def _detect_time_range(
        self,
        *,
        ds: Any = None,
        strict: bool = False,
    ) -> TimeRange | None:
        """Detect the temporal range of the dataset if applicable.

        This method should be overridden in subclasses.

        Parameters
        ----------
        ds: Any, optional
            If provided, use this dataset to detect the time range. If None, the
            dataset will be fetched according to the settings in the DataSource.

        Returns
        -------
        range: TimeRange, optional
            Instance containing the start and end of the time dimension. Range is
            inclusive on both sides. None if no time dimension could be found.
        """
        msg = (
            f"Source of type {type(self)} does not support detecting temporal extents."
        )
        if strict:
            raise NotImplementedError(msg)
        else:
            logger.warning(msg + " skipping...")
        return None

    def _detect_bbox(self, *, strict: bool = False) -> TotalBounds | None:
        """Detect the bounding box and crs of the dataset if applicable.

        This method should be overridden in subclasses.

        Returns
        -------
        bbox: TotalBounds, optional
            The bounding box coordinates of the data as (bbox, crs), where bbox is
            (minx, miny, maxx, maxy) and crs is the EPSG code of the CRS of the
            coordinates returned in bbox. None if not applicable.
        """
        msg = f"Source of type {type(self)} does not support detecting spatial extents."
        if strict:
            raise NotImplementedError(msg)
        else:
            logger.warning(msg + " skipping...")
        return None


def _abs_path(root: Union[Path, str], rel_path: Union[Path, str]) -> str:
    path = Path(str(rel_path))
    if not path.is_absolute():
        if root is not None:
            rel_path = join(root, rel_path)
        path = Path(abspath(rel_path))
    return str(path)
