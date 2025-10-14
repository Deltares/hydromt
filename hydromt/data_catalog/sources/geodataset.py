"""DataSource class for the GeoDataset type."""

import logging
from os.path import basename, splitext
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

import xarray as xr
from pydantic import Field
from pyproj.exceptions import CRSError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from pystac import MediaType

from hydromt.data_catalog.adapters.geodataset import GeoDatasetAdapter
from hydromt.data_catalog.drivers.geodataset.geodataset_driver import GeoDatasetDriver
from hydromt.data_catalog.sources.data_source import DataSource
from hydromt.error import NoDataStrategy
from hydromt.gis.gis_utils import _parse_geom_bbox_buffer
from hydromt.typing import (
    Bbox,
    Geom,
    StrPath,
    TimeRange,
    TotalBounds,
)
from hydromt.typing.fsspec_types import FSSpecFileSystem
from hydromt.typing.type_def import GeomBuffer, Predicate

logger = logging.getLogger(__name__)


class GeoDatasetSource(DataSource):
    """DataSource class for the GeoDatasetSource type."""

    data_type: ClassVar[Literal["GeoDataset"]] = "GeoDataset"
    _fallback_driver_read: ClassVar[str] = "geodataset_vector"
    _fallback_driver_write: ClassVar[str] = "geodataset_xarray"

    driver: GeoDatasetDriver
    data_adapter: GeoDatasetAdapter = Field(default_factory=GeoDatasetAdapter)

    def read_data(
        self,
        *,
        mask: Optional[Geom] = None,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        single_var_as_array: bool = True,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> Optional[Union[xr.Dataset, xr.DataArray]]:
        """
        Read data from this source.

        Args:
        """
        self._mark_as_used()
        self._log_start_read_data()

        # Transform time_range and variables to match the data source
        tr = self.data_adapter._to_source_timerange(time_range)
        vrs = self.data_adapter._to_source_variables(variables)

        uris: List[str] = self.uri_resolver.resolve(
            self.full_uri,
            time_range=tr,
            variables=vrs,
            metadata=self.metadata,
            handle_nodata=handle_nodata,
        )

        ds: Optional[xr.Dataset] = self.driver.read(
            uris,
            time_range=tr,
            variables=vrs,
            metadata=self.metadata,
            handle_nodata=handle_nodata,
        )
        return self.data_adapter.transform(
            ds,
            self.metadata,
            predicate=predicate,
            single_var_as_array=single_var_as_array,
            mask=mask,
            variables=variables,
            time_range=time_range,
            handle_nodata=handle_nodata,
        )

    def to_file(
        self,
        file_path: StrPath,
        *,
        driver_override: Optional[GeoDatasetDriver] = None,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: GeomBuffer = 0,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        single_var_as_array: bool = True,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
    ) -> Optional["GeoDatasetSource"]:
        """
        Write the GeoDatasetSource to a local file.

        args:
        """
        if driver_override is None and not self.driver.supports_writing:
            # default to fallback driver.
            driver: GeoDatasetDriver = GeoDatasetDriver.model_validate(
                self._fallback_driver_write
            )
        elif driver_override:
            if not driver_override.supports_writing:
                raise RuntimeError(
                    f"driver: '{driver_override.name}' does not support writing data."
                )
            driver: GeoDatasetDriver = driver_override
        else:
            # use local filesystem
            driver: GeoDatasetDriver = self.driver.model_copy(
                update={"filesystem": FSSpecFileSystem.create("local")}
            )

        if bbox is not None or (mask is not None and buffer > 0):
            mask = _parse_geom_bbox_buffer(mask, bbox, buffer)
        ds: Optional[Union[xr.Dataset, xr.DataArray]] = self.read_data(
            mask=mask,
            predicate=predicate,
            variables=variables,
            single_var_as_array=single_var_as_array,
            time_range=time_range,
            handle_nodata=handle_nodata,
        )
        if ds is None:  # handle_nodata == ignore
            return None

        dest_path: str = driver.write(
            file_path,
            ds,
            **kwargs,
        )

        # update driver based on local path
        update: Dict[str, Any] = {"uri": dest_path, "root": None, "driver": driver}

        return self.model_copy(update=update)

    def _detect_bbox(
        self,
        *,
        strict: bool = False,
        ds: Optional[xr.Dataset] = None,
    ) -> TotalBounds:
        """Detect the bounding box and crs of the dataset.

        If no dataset is provided, it will be fetched according to the settings in the
        adapter. also see :py:meth:`hydromt.GeoDatasetAdapter.get_data`. the
        coordinates are in the CRS of the dataset itself, which is also returned
        alongside the coordinates.


        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the bounding box of.
            If none is provided, :py:meth:`hydromt.GeoDatasetAdapter.get_data`
            will be used to fetch the it before detecting.

        Returns
        -------
        bbox: Tuple[np.float64,np.float64,np.float64,np.float64]
            the bounding box coordinates of the data. coordinates are returned as
            [xmin,ymin,xmax,ymax]
        crs: int
            The ESPG code of the CRS of the coordinates returned in bbox
        """
        if ds is None:
            ds = self.read_data()
        crs = ds.vector.crs.to_epsg()
        bounds = ds.vector.bounds

        return bounds, crs

    def _detect_time_range(
        self,
        *,
        strict: bool = False,
        ds: Optional[xr.Dataset] = None,
    ) -> TimeRange | None:
        """Detect the temporal range of the dataset.

        If no dataset is provided, it will be fetched accodring to the settings in the
        addapter. also see :py:meth:`hydromt.GeoDatasetAdapter.get_data`.

        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the time range of. It must have a time dimentsion set.
            If none is provided, :py:meth:`hydromt.GeoDatasetAdapter.get_data`
            will be used to fetch the it before detecting.

        Returns
        -------
        range: TimeRange
            A TimeRange containing the start and end of the time dimension. Range is
            inclusive on both sides.
        """
        if ds is None:
            ds = self.read_data()
        return TimeRange(
            start=ds[ds.vector.time_dim].min().values,
            end=ds[ds.vector.time_dim].max().values,
        )

    def to_stac_catalog(
        self,
        handle_nodata: NoDataStrategy = NoDataStrategy.IGNORE,
    ) -> Optional[StacCatalog]:
        """
        Convert a geodataset into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - handle_nodata (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "IGNORE" to skip
          the dataset on failure

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataset, or
          None if the dataset was skipped.
        """
        try:
            bbox, crs = self.get_bbox(detect=True)
            bbox = list(bbox)
            time_range = self.get_time_range(detect=True)
            props = {**self.metadata.model_dump(exclude_none=True), "crs": crs}
            ext = splitext(self.uri)[-1]
            if ext in [".nc", ".vrt"]:
                media_type = MediaType.HDF5
            else:
                raise RuntimeError(
                    f"Unknown extension: {ext} cannot determine media type"
                )
        except (IndexError, KeyError, CRSError) as e:
            if handle_nodata == NoDataStrategy.IGNORE:
                logger.warning(
                    "Skipping {name} during stac conversion because"
                    "because detecting spacial extent failed."
                )
                return
            else:
                raise e

        stac_catalog = StacCatalog(
            self.name,
            description=self.name,
        )
        stac_item = StacItem(
            self.name,
            geometry=None,
            bbox=bbox,
            properties=props,
            datetime=None,
            start_datetime=time_range.start,
            end_datetime=time_range.end,
        )
        stac_asset = StacAsset(str(self.uri), media_type=media_type)
        base_name = basename(self.uri)
        stac_item.add_asset(base_name, stac_asset)

        stac_catalog.add_item(stac_item)
        return stac_catalog

    @classmethod
    def _infer_default_driver(
        cls,
        uri: str | None = None,
    ) -> str:
        return super()._infer_default_driver(uri, GeoDatasetDriver)
