"""DataSource class for the RasterDataset type."""
from datetime import datetime
from logging import Logger, getLogger
from os.path import basename, splitext
from typing import Any, ClassVar, Dict, Literal, Optional, cast

import pandas as pd
import xarray as xr
from pydantic import ValidationInfo, field_validator
from pyproj import CRS
from pyproj.exceptions import CRSError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from pystac import MediaType

from hydromt._typing import Bbox, ErrorHandleMethod, Geom, TimeRange, TotalBounds
from hydromt.data_adapter.rasterdataset import RasterDatasetAdapter
from hydromt.data_source.data_source import DataSource
from hydromt.drivers.rasterdataset_driver import RasterDatasetDriver
from hydromt.drivers.zarr_driver import ZarrDriver

_KNOWN_DRIVERS: Dict[str, RasterDatasetDriver] = {"zarr": ZarrDriver}
logger: Logger = getLogger(__name__)


def driver_from_str(driver_str: str, **kwargs) -> RasterDatasetDriver:
    """Construct RasterDatasetDriver."""
    if driver_str not in _KNOWN_DRIVERS.keys():
        raise ValueError(
            f"driver {driver_str} not in known RasterDatasetDrivers: {_KNOWN_DRIVERS.keys()}"
        )

    return _KNOWN_DRIVERS[driver_str](**kwargs)


class RasterDatasetSource(DataSource):
    """DataSource class for the RasterDataset type."""

    data_type: ClassVar[Literal["RasterDataset"]] = "RasterDataset"
    driver: RasterDatasetDriver
    data_adapter: RasterDatasetAdapter

    @field_validator("driver", mode="before")
    @classmethod
    def _check_geodataframe_drivers(cls, v: Any, info: ValidationInfo) -> str:
        if isinstance(v, str):
            if v not in _KNOWN_DRIVERS:
                raise ValueError(f"unknown driver '{v}'")
            return driver_from_str(v, **info.data.get("driver_kwargs"))
        elif hasattr(v, "read"):  # driver duck-typing
            return v
        else:
            raise ValueError(f"unknown driver type: {str(v)}")

    zoom_levels: Optional[Dict[int, float]] = None

    def read_data(
        self,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0,
        predicate: str = "intersects",
        timerange: Optional[TimeRange] = None,
        zoom_level: int = 0,
        logger: Optional[Logger] = None,
    ) -> xr.Dataset:
        """
        Read data from this source.

        Data is returned piecewise as a generator, so that users of this class can
        filter the data before taking it all into memory.

        Args:
        """
        ds: xr.Dataset = self.driver.read(
            self.uri,
            bbox=bbox,
            mask=mask,
            buffer=buffer,
            crs=self.crs,
            predicate=predicate,
            timerange=timerange,
            zoom_level=zoom_level,
            **self.driver_kwargs,
        )
        return self.data_adapter.transform(
            ds,
            bbox=bbox,
            mask=mask,
            buffer=buffer,
            crs=self.crs,
            predicate=predicate,
            timerange=timerange,
            zoom_level=zoom_level,
        )

    def get_bbox(self, crs: Optional[CRS], detect: bool = True) -> TotalBounds:
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
        bbox = self.data_adapter.harmonization_settings.extent.get("bbox", None)
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
        time_range = self.data_adapter.harmonization_settings.extent.get(
            "time_range", None
        )
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
        on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE,
    ) -> Optional[StacCatalog]:
        """
        Convert a rasterdataset into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - on_error (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "skip" to skip the
          dataset on failure, and "coerce" (default) to set default values on failure.

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
            props = {**self.harmonization.meta, "crs": crs}
            ext = splitext(self.uri)[-1]
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
                    f"Unknown extention: {ext} cannot determine media type"
                )
        except (IndexError, KeyError, CRSError) as e:
            if on_error == ErrorHandleMethod.SKIP:
                logger.warning(
                    "Skipping {name} during stac conversion because"
                    "because detecting spacial extent failed."
                )
                return
            elif on_error == ErrorHandleMethod.COERCE:
                bbox = [0.0, 0.0, 0.0, 0.0]
                props = self.harmonization.meta
                start_dt = datetime(1, 1, 1)
                end_dt = datetime(1, 1, 1)
                media_type = MediaType.JSON
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
            stac_asset = StacAsset(str(self.uri), media_type=media_type)
            base_name = basename(self.uri)
            stac_item.add_asset(base_name, stac_asset)

            stac_catalog.add_item(stac_item)
            return stac_catalog
