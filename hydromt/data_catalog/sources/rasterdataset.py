"""DataSource class for the RasterDataset type."""

from logging import Logger, getLogger
from os.path import basename, splitext
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union, cast

import pandas as pd
import xarray as xr
from fsspec import filesystem
from pydantic import Field
from pyproj import CRS
from pyproj.exceptions import CRSError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from pystac import MediaType

from hydromt._typing import (
    Bbox,
    Geom,
    NoDataStrategy,
    StrPath,
    TimeRange,
    TotalBounds,
    Zoom,
)
from hydromt.data_catalog.adapters.rasterdataset import RasterDatasetAdapter
from hydromt.data_catalog.drivers import RasterDatasetDriver
from hydromt.data_catalog.sources.data_source import DataSource
from hydromt.gis._gis_utils import _parse_geom_bbox_buffer

logger: Logger = getLogger(__name__)


class RasterDatasetSource(DataSource):
    """DataSource class for the RasterDataset type."""

    data_type: ClassVar[Literal["RasterDataset"]] = "RasterDataset"
    _fallback_driver_read: ClassVar[str] = "rasterio"
    _fallback_driver_write: ClassVar[str] = "raster_xarray"
    driver: RasterDatasetDriver
    data_adapter: RasterDatasetAdapter = Field(default_factory=RasterDatasetAdapter)

    def read_data(
        self,
        *,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0,
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        zoom: Optional[Zoom] = None,
        chunks: Optional[dict] = None,
        single_var_as_array: bool = True,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Read data from this source.

        Args:
        """
        self._used = True
        if bbox is not None or (mask is not None and buffer > 0):
            mask = _parse_geom_bbox_buffer(mask, bbox, buffer)

        # Transform time_range and variables to match the data source
        tr = self.data_adapter._to_source_timerange(time_range)
        vrs = self.data_adapter._to_source_variables(variables)

        uris: List[str] = self.uri_resolver.resolve(
            self.full_uri,
            time_range=tr,
            mask=mask,
            variables=vrs,
            zoom=zoom,
            metadata=self.metadata,
            handle_nodata=handle_nodata,
        )

        ds: xr.Dataset = self.driver.read(
            uris,
            mask=mask,
            time_range=tr,
            variables=vrs,
            zoom=zoom,
            chunks=chunks,
            metadata=self.metadata,
            handle_nodata=handle_nodata,
        )
        return self.data_adapter.transform(
            ds,
            self.metadata,
            mask=mask,
            variables=variables,
            time_range=time_range,
            single_var_as_array=single_var_as_array,
        )

    def to_file(
        self,
        file_path: StrPath,
        *,
        driver_override: Optional[RasterDatasetDriver] = None,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0.0,
        time_range: Optional[TimeRange] = None,
        zoom: Optional[Zoom] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
    ) -> "RasterDatasetSource":
        """
        Write the RasterDatasetSource to a local file.

        args:
        """
        if driver_override is None and not self.driver.supports_writing:
            # default to fallback driver
            driver: RasterDatasetDriver = RasterDatasetDriver.model_validate(
                self._fallback_driver_write
            )
        elif driver_override:
            if not driver_override.supports_writing:
                raise RuntimeError(
                    f"driver: '{driver_override.name}' does not support writing data."
                )
            driver: RasterDatasetDriver = driver_override
        else:
            # use local filesystem
            driver: RasterDatasetDriver = self.driver.model_copy(
                update={"filesystem": filesystem("local")}
            )

        ds: Optional[xr.Dataset] = self.read_data(
            bbox=bbox,
            mask=mask,
            buffer=buffer,
            time_range=time_range,
            zoom=zoom,
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

    def get_bbox(self, crs: Optional[CRS] = None, detect: bool = True) -> TotalBounds:
        """Return the bounding box and espg code of the dataset.

        if the bounding box is not set and detect is True,
        :py:meth:`hydromt.RasterdatasetAdapter.detect_bbox` will be used to detect it.

        Parameters
        ----------
        detect: bool, Optional
            whether to detect the bounding box if it is not set. If False, and it's not
            set None will be returned.

        Returns
        -------
        bbox: Tuple[np.float64,np.float64,np.float64,np.float64]
            the bounding box coordinates of the data. coordinates are returned as
            [xmin,ymin,xmax,ymax]
        crs: int
            The ESPG code of the CRS of the coordinates returned in bbox
        """
        bbox = self.metadata.extent.get("bbox", None)
        crs = cast(int, crs)
        if bbox is None and detect:
            bbox, crs = self.detect_bbox()

        return bbox, crs

    def get_time_range(
        self,
        detect: bool = True,
    ) -> TimeRange:
        """Detect the time range of the dataset.

        if the time range is not set and detect is True,
        :py:meth:`hydromt.RasterdatasetAdapter.detect_time_range` will be used
        to detect it.


        Parameters
        ----------
        detect: bool, Optional
            whether to detect the time range if it is not set. If False, and it's not
            set None will be returned.

        Returns
        -------
        range: Tuple[np.datetime64, np.datetime64]
            A tuple containing the start and end of the time dimension. Range is
            inclusive on both sides.
        """
        time_range = self.metadata.extent.get("time_range", None)
        if time_range is None and detect:
            time_range = self.detect_time_range()

        return time_range

    def detect_bbox(
        self,
        ds: Optional[xr.Dataset] = None,
    ) -> TotalBounds:
        """Detect the bounding box and crs of the dataset.

        If no dataset is provided, it will be fetched according to the settings in the
        adapter. also see :py:meth:`hydromt.RasterdatasetAdapter.get_data`. the
        coordinates are in the CRS of the dataset itself, which is also returned
        alongside the coordinates.


        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the bounding box of.
            If none is provided, :py:meth:`hydromt.RasterdatasetAdapter.get_data`
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
        crs = ds.raster.crs.to_epsg()
        bounds = ds.raster.bounds

        return bounds, crs

    def detect_time_range(
        self,
        ds: Optional[xr.Dataset] = None,
    ) -> TimeRange:
        """Detect the temporal range of the dataset.

        If no dataset is provided, it will be fetched accodring to the settings in the
        addapter. also see :py:meth:`hydromt.RasterdatasetAdapter.get_data`.

        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the time range of. It must have a time dimentsion set.
            If none is provided, :py:meth:`hydromt.RasterdatasetAdapter.get_data`
            will be used to fetch the it before detecting.

        Returns
        -------
        range: Tuple[np.datetime64, np.datetime64]
            A tuple containing the start and end of the time dimension. Range is
            inclusive on both sides.
        """
        if ds is None:
            ds = self.read_data()
        return (
            ds[ds.raster.time_dim].min().values,
            ds[ds.raster.time_dim].max().values,
        )

    def to_stac_catalog(
        self,
        handle_nodata: NoDataStrategy = NoDataStrategy.IGNORE,
    ) -> Optional[StacCatalog]:
        """
        Convert a rasterdataset into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - handle_nodata (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "ignore" to skip the
          dataset on failure

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataset, or None
          if the dataset was skipped.
        """
        try:
            bbox, crs = self.get_bbox(detect=True)
            bbox = list(bbox)
            start_dt, end_dt = self.get_time_range(detect=True)
            start_dt = pd.to_datetime(start_dt)
            end_dt = pd.to_datetime(end_dt)
            props = {**self.metadata.model_dump(exclude_none=True), "crs": crs}
            ext = splitext(self.full_uri)[-1]
            if ext == ".nc" or ext == ".vrt":
                media_type = MediaType.HDF5
            elif ext == ".tiff":
                media_type = MediaType.TIFF
            elif ext == ".cog":
                media_type = MediaType.COG
            elif ext == ".png":
                media_type = MediaType.PNG
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

        else:
            # else makes type checkers a bit happier
            stac_catalog = StacCatalog(
                self.name,
                description=self.name,
            )
            stac_item = StacItem(
                self.name,
                geometry=None,
                bbox=list(bbox),
                properties=props,
                datetime=None,
                start_datetime=start_dt,
                end_datetime=end_dt,
            )
            stac_asset = StacAsset(str(self.full_uri), media_type=media_type)
            base_name = basename(self.full_uri)
            stac_item.add_asset(base_name, stac_asset)

            stac_catalog.add_item(stac_item)
            return stac_catalog
