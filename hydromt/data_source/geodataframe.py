"""Generic DataSource for GeoDataFrames."""

from datetime import datetime
from logging import Logger, getLogger
from os.path import basename, splitext
from typing import Any, ClassVar, Dict, List, Literal, Optional

import geopandas as gpd
from pydantic import ValidationInfo, field_validator
from pyproj import CRS
from pyproj.exceptions import CRSError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from pystac import MediaType

from hydromt._typing import Bbox, ErrorHandleMethod, Geom, NoDataStrategy, TotalBounds
from hydromt.data_adapter.geodataframe import GeoDataFrameAdapter
from hydromt.driver.geodataframe_driver import GeoDataFrameDriver
from hydromt.driver.pyogrio_driver import PyogrioDriver

from .data_source import DataSource

logger: Logger = getLogger(__name__)

# placeholder for proper plugin behaviour later on.
_KNOWN_DRIVERS: Dict[str, GeoDataFrameDriver] = {"pyogrio": PyogrioDriver}


def driver_from_str(driver_str: str, **driver_kwargs) -> GeoDataFrameDriver:
    """Construct GeoDataFrame driver."""
    if driver_str not in _KNOWN_DRIVERS.keys():
        raise ValueError(
            f"driver {driver_str} not in known GeoDataFrameDrivers: {_KNOWN_DRIVERS.keys()}"
        )

    return _KNOWN_DRIVERS[driver_str](**driver_kwargs)


class GeoDataFrameSource(DataSource):
    """
    DataSource for GeoDataFrames.

    Reads and validates DataCatalog entries.
    """

    data_type: ClassVar[Literal["GeoDataFrame"]] = "GeoDataFrame"
    driver: GeoDataFrameDriver
    data_adapter: GeoDataFrameAdapter

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

    def read_data(
        self,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0.0,
        variables: Optional[List[str]] = None,
        predicate: str = "intersects",
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
    ) -> gpd.GeoDataFrame:
        """Use the driver and data adapter to read and harmonize the data."""
        gdf: gpd.GeoDataFrame = self.driver.read(
            self.uri,
            bbox=bbox,
            geom=mask,
            buffer=buffer,
            crs=self.crs,
            predicate=predicate,
            variables=variables,
            handle_nodata=handle_nodata,
            logger=logger,
            **self.driver_kwargs,
        )
        return self.data_adapter.transform(
            gdf,
            bbox=bbox,
            mask=mask,
            buffer=buffer,
            crs=self.crs,
            predicate=predicate,
            variables=variables,
            handle_nodata=handle_nodata,
            logger=logger,
        )

    def get_bbox(self, crs: Optional[CRS], detect: bool = True) -> TotalBounds:
        """Return the bounding box and espg code of the dataset.

        if the bounding box is not set and detect is True,
        :py:meth:`hydromt.GeoDataframeAdapter.detect_bbox` will be used to detect it.

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
        if bbox is None and detect:
            bbox, crs = self.detect_bbox()

        return bbox, crs

    def detect_bbox(
        self,
        gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> TotalBounds:
        """Detect the bounding box and crs of the dataset.

        If no dataset is provided, it will be fetched acodring to the settings in the
        adapter. also see :py:meth:`hydromt.GeoDataframeAdapter.get_data`. the
        coordinates are in the CRS of the dataset itself, which is also returned
        alongside the coordinates.


        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the bounding box of.
            If none is provided, :py:meth:`hydromt.GeoDataframeAdapter.get_data`
            will be used to fetch the it before detecting.

        Returns
        -------
        bbox: Tuple[np.float64,np.float64,np.float64,np.float64]
            the bounding box coordinates of the data. coordinates are returned as
            [xmin,ymin,xmax,ymax]
        crs: int
            The ESPG code of the CRS of the coordinates returned in bbox
        """
        if gdf is None:
            gdf = self.read_data()

        crs = gdf.geometry.crs.to_epsg()
        bounds = gdf.geometry.total_bounds
        return bounds, crs

    def to_stac_catalog(
        self,
        on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE,
    ) -> Optional[StacCatalog]:
        """
        Convert a geodataframe into a STAC Catalog representation.

        Since geodataframes don't support temporal dimension the `datetime`
        property will always be set to 0001-01-01. The collection will contain an
        asset for each of the associated files.


        Parameters
        ----------
        - on_error (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "skip" to skip
          the dataset on failure, and "coerce" (default) to set
          default values on failure.

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataset, or
          None if the dataset was skipped.
        """
        try:
            bbox, crs = self.data_adapter.get_bbox(detect=True)  # Should move to driver
            bbox = list(bbox)
            props = {**self.harmonization.meta, "crs": crs}
            ext = splitext(self.uri)[-1]
            if ext == ".gpkg":
                media_type = MediaType.GEOPACKAGE
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
                media_type = MediaType.JSON
            else:
                raise e
        else:
            stac_catalog = StacCatalog(
                self.name,
                description=self.name,
            )
            stac_item = StacItem(
                self.name,
                geometry=None,
                bbox=list(bbox),
                properties=props,
                datetime=datetime(1, 1, 1),
            )
            stac_asset = StacAsset(str(self.uri), media_type=media_type)
            base_name = basename(self.uri)
            stac_item.add_asset(base_name, stac_asset)

            stac_catalog.add_item(stac_item)
            return stac_catalog
