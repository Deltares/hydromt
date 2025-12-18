"""DataSource class for the RasterDataset type."""

import logging
from os.path import basename, splitext
from pathlib import Path
from typing import Any, ClassVar, List, Literal, Optional

import numpy as np
import xarray as xr
from pydantic import Field
from pyproj.exceptions import CRSError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from pystac import MediaType

from hydromt.data_catalog.adapters.rasterdataset import RasterDatasetAdapter
from hydromt.data_catalog.drivers import RasterDatasetDriver
from hydromt.data_catalog.sources.data_source import DataSource
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.gis.gis_utils import _parse_geom_bbox_buffer
from hydromt.typing import (
    Bbox,
    Geom,
    TimeRange,
    TotalBounds,
    Zoom,
)
from hydromt.typing.fsspec_types import FSSpecFileSystem

logger = logging.getLogger(__name__)


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
        buffer: int = 0,
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        zoom: Optional[Zoom] = None,
        chunks: Optional[dict] = None,
        single_var_as_array: bool = True,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> xr.Dataset | xr.DataArray | None:
        """
        Read data from this source.

        Args:
        """
        self._mark_as_used()
        self._log_start_read_data()

        if bbox is not None:
            # buffer will be applied in transform
            mask = _parse_geom_bbox_buffer(mask, bbox)

        # Transform time_range and variables to match the data source
        src_time_range = self.data_adapter._to_source_timerange(time_range)
        vrs = self.data_adapter._to_source_variables(variables)

        uris: List[str] = self.uri_resolver.resolve(
            self.full_uri,
            time_range=src_time_range,
            mask=mask,
            variables=vrs,
            zoom=zoom,
            metadata=self.metadata,
            handle_nodata=handle_nodata,
        )
        if not uris:
            return None  # handle_nodata == ignore

        ds: xr.Dataset = self.driver.read(
            uris,
            handle_nodata=handle_nodata,
            mask=mask,
            variables=vrs,
            zoom=zoom,
            chunks=chunks,
            metadata=self.metadata,
        )
        if ds is None:
            return None  # handle_nodata == ignore

        return self.data_adapter.transform(
            ds,
            self.metadata,
            mask=mask,
            variables=variables,
            time_range=time_range,
            single_var_as_array=single_var_as_array,
            buffer=buffer,
            handle_nodata=handle_nodata,
        )

    def to_file(
        self,
        file_path: Path | str,
        *,
        driver_override: RasterDatasetDriver | None = None,
        bbox: Bbox | None = None,
        mask: Geom | None = None,
        buffer: int = 0,
        time_range: TimeRange | None = None,
        zoom: Zoom | None = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        write_kwargs: dict[str, Any] | None = None,
        variables: List[str] | None = None,
    ) -> "RasterDatasetSource | None":
        """
        Write the RasterDatasetSource to a local file.

        args:
        """
        if driver_override is None and not self.driver.supports_writing:
            # default to fallback driver
            driver = RasterDatasetDriver.model_validate(self._fallback_driver_write)
        elif driver_override:
            if not driver_override.supports_writing:
                raise RuntimeError(
                    f"driver: '{driver_override.name}' does not support writing data."
                )
            driver = driver_override
        else:
            # use local filesystem
            driver = self.driver.model_copy(
                update={"filesystem": FSSpecFileSystem.create("local")}
            )

        ds = self.read_data(
            bbox=bbox,
            mask=mask,
            buffer=buffer,
            time_range=time_range,
            zoom=zoom,
            handle_nodata=handle_nodata,
            variables=variables,
        )
        if ds is None:  # handle_nodata == ignore
            return None

        file_path = Path(file_path)

        dest_path = driver.write(
            file_path,
            ds,
            write_kwargs=write_kwargs,
        )

        # update driver based on local path
        update = {
            "uri": dest_path.as_posix(),
            "root": None,
            "driver": driver,
        }

        return self.model_copy(update=update)

    def _detect_bbox(
        self,
        *,
        strict: bool = False,
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

    def _detect_time_range(
        self,
        *,
        strict: bool = False,
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
        range: TimeRange
            A tuple containing the start and end of the time dimension. Range is
            inclusive on both sides.
        """
        if ds is None:
            ds = self.read_data()

        if not np.issubdtype(ds[ds.raster.time_dim].dtype, np.datetime64):
            ds = ds.convert_calendar("standard")

        return TimeRange(
            start=ds[ds.raster.time_dim].values.min(),
            end=ds[ds.raster.time_dim].values.max(),
        )

    def to_stac_catalog(
        self,
        handle_nodata: NoDataStrategy = NoDataStrategy.WARN,
    ) -> Optional[StacCatalog]:
        """
        Convert a rasterdataset into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - handle_nodata (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "warn" to log a warning, "ignore" to skip the
          dataset on failure

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataset, or None
          if the dataset was skipped.
        """
        try:
            bbox, crs = self.get_bbox(detect=True)
            bbox = list(bbox)
            time_range = self.get_time_range(detect=True)
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
        except (IndexError, KeyError, CRSError):
            exec_nodata_strat(
                f"Skipping {self.name} during stac conversion because detecting spacial extent failed.",
                strategy=handle_nodata,
            )
            return None

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
                start_datetime=time_range.start,
                end_datetime=time_range.end,
            )
            stac_asset = StacAsset(str(self.full_uri), media_type=media_type)
            base_name = basename(self.full_uri)
            stac_item.add_asset(base_name, stac_asset)

            stac_catalog.add_item(stac_item)
            return stac_catalog

    @classmethod
    def _infer_default_driver(
        cls,
        uri: str | None = None,
    ) -> str:
        return super()._infer_default_driver(uri, RasterDatasetDriver)
