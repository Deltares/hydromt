"""DataSource class for the GeoDataFrame type."""

from datetime import datetime
from logging import Logger, getLogger
from os.path import basename, splitext
from typing import Any, ClassVar, Dict, List, Literal, Optional

import geopandas as gpd
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
    ErrorHandleMethod,
    Geom,
    NoDataStrategy,
    StrPath,
    TotalBounds,
)
from hydromt.data_catalog.adapters.geodataframe import GeoDataFrameAdapter
from hydromt.data_catalog.drivers import GeoDataFrameDriver
from hydromt.gis.gis_utils import parse_geom_bbox_buffer

from .data_source import DataSource

logger: Logger = getLogger(__name__)


class GeoDataFrameSource(DataSource):
    """
    DataSource for GeoDataFrames.

    Reads and validates DataCatalog entries.
    """

    data_type: ClassVar[Literal["GeoDataFrame"]] = "GeoDataFrame"
    _fallback_driver_read: ClassVar[str] = "pyogrio"
    _fallback_driver_write: ClassVar[str] = "pyogrio"
    driver: GeoDataFrameDriver
    data_adapter: GeoDataFrameAdapter = Field(default_factory=GeoDataFrameAdapter)

    def read_data(
        self,
        *,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0.0,
        variables: Optional[List[str]] = None,
        predicate: str = "intersects",
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
    ) -> Optional[gpd.GeoDataFrame]:
        """Use the driver and data adapter to read and harmonize the data."""
        self._used = True
        if bbox is not None or (mask is not None and buffer > 0):
            mask = parse_geom_bbox_buffer(mask, bbox, buffer)
        gdf: gpd.GeoDataFrame = self.driver.read(
            self.full_uri,
            mask=mask,
            predicate=predicate,
            variables=variables,
            metadata=self.metadata,
            handle_nodata=handle_nodata,
            logger=logger,
        )
        return self.data_adapter.transform(
            gdf,
            self.metadata,
            mask=mask,
            predicate=predicate,
            variables=variables,
            handle_nodata=handle_nodata,
            logger=logger,
        )

    def to_file(
        self,
        file_path: StrPath,
        *,
        driver_override: Optional[GeoDataFrameDriver] = None,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0.0,
        variables: Optional[List[str]] = None,
        predicate: str = "intersects",
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
        **kwargs,
    ) -> "GeoDataFrameSource":
        """
        Write the GeoDataFrameSource to a local file.

        args:
        """
        if driver_override is None and not self.driver.supports_writing:
            # default to fallback driver
            driver: GeoDataFrameDriver = GeoDataFrameDriver.model_validate(
                self._fallback_driver_write
            )
        elif driver_override:
            if not driver_override.supports_writing:
                raise RuntimeError(
                    f"driver: '{driver_override.name}' does not support writing data."
                )
            driver: GeoDataFrameDriver = driver_override
        else:
            # use local filesystem
            driver: GeoDataFrameDriver = self.driver.model_copy(
                update={"filesystem": filesystem("local")}
            )

        gdf: Optional[gpd.GeoDataFrame] = self.read_data(
            bbox=bbox,
            mask=mask,
            buffer=buffer,
            variables=variables,
            predicate=predicate,
            handle_nodata=handle_nodata,
            logger=logger,
        )
        if gdf is None:  # handle_nodata == ignore
            return None

        # update source and its driver based on local path
        update: Dict[str, Any] = {"uri": file_path, "root": None, "driver": driver}

        driver.write(
            file_path,
            gdf,
            **kwargs,
        )

        return self.model_copy(update=update)

    def get_bbox(self, crs: Optional[CRS] = None, detect: bool = True) -> TotalBounds:
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
        bbox = self.metadata.extent.get("bbox", None)
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
            bbox, crs = self.get_bbox(detect=True)  # Should move to driver
            bbox = list(bbox)
            props = {**self.metadata.model_dump(), "crs": crs}
            ext = splitext(self.full_uri)[-1]
            if ext == ".gpkg":
                media_type = MediaType.GEOPACKAGE
            else:
                raise RuntimeError(
                    f"Unknown extension: {ext} cannot determine media type"
                )
        except (IndexError, KeyError, CRSError):
            if on_error == ErrorHandleMethod.SKIP:
                logger.warning(
                    "Skipping {name} during stac conversion because"
                    "because detecting spacial extent failed."
                )
                return
            elif on_error == ErrorHandleMethod.COERCE:
                bbox = [0.0, 0.0, 0.0, 0.0]
                props = self.data_adapter.meta
                media_type = MediaType.JSON
            else:
                raise
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
            stac_asset = StacAsset(str(self.full_uri), media_type=media_type)
            base_name = basename(self.full_uri)
            stac_item.add_asset(base_name, stac_asset)

            stac_catalog.add_item(stac_item)
            return stac_catalog
