"""The GeoDataFrame adapter performs transformations on GeoDataFrames."""
from datetime import datetime
from logging import Logger, getLogger
from os.path import basename, splitext
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pyproj
from pydantic import PrivateAttr
from pyproj import CRS
from pyproj.exceptions import CRSError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from pystac import MediaType

from hydromt._typing import (
    ErrorHandleMethod,
    Geom,
    NoDataException,
    NoDataStrategy,
    TotalBounds,
    _exec_nodata_strat,
)
from hydromt.data_sources.geodataframe import GeoDataSource
from hydromt.gis import parse_geom_bbox_buffer, utils

from .data_adapter_base import DataAdapterBase

logger: Logger = getLogger(__name__)


class GeoDataFrameAdapter(DataAdapterBase):
    """The GeoDataFrameAdapter performs transformations on GeoDataFrames."""

    source: GeoDataSource
    _used: bool = PrivateAttr(False)

    def get_data(
        self,
        bbox: Optional[List[float]] = None,
        mask: Optional[gpd.GeoDataFrame] = None,
        buffer: float = 0.0,
        variables: Optional[List[str]] = None,
        predicate: str = "intersects",
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
    ) -> Optional[gpd.GeoDataFrame]:
        """Read in data and transform them to HydroMT standards."""
        self._used = True  # mark used
        try:
            gdf: gpd.GeoDataFrame = self.source.read_data(
                bbox, mask, buffer, variables, predicate, handle_nodata, logger=logger
            )
        except NoDataException:
            _exec_nodata_strat(
                f"No data was read from source: {self.name}",
                strategy=handle_nodata,
                logger=logger,
            )
            return None

        # rename variables and parse crs & nodata
        gdf = self._rename_vars(gdf)
        gdf = self._set_crs(gdf, logger=logger)
        gdf = self._set_nodata(gdf)
        # slice
        gdf = GeoDataFrameAdapter._slice_data(
            gdf,
            variables,
            mask,
            bbox,
            self.source.crs,
            buffer,
            predicate,
            handle_nodata,
            logger=logger,
        )
        # uniformize
        gdf = self._apply_unit_conversions(gdf, logger=logger)
        gdf = self._set_metadata(gdf)
        return gdf

    def _rename_vars(self, gdf: gpd.GeoDataFrame):
        # rename and select columns
        if rename := self.source.rename:
            rename = {k: v for k, v in rename.items() if k in gdf.columns}
            gdf = gdf.rename(columns=rename)
        return gdf

    def _set_crs(self, gdf: gpd.GeoDataFrame, logger=logger):
        if crs := self.source.crs is not None and gdf.crs is None:
            gdf.set_crs(crs, inplace=True)
        elif gdf.crs is None:
            raise ValueError(
                f"GeoDataFrame {self.source.name}: CRS not defined in data catalog or data."
            )
        elif self.source.crs is not None and gdf.crs != pyproj.CRS.from_user_input(
            self.source.crs
        ):
            logger.warning(
                f"GeoDataFrame {self.source.name}: CRS from data catalog does not match CRS of"
                " data. The original CRS will be used. Please check your data catalog."
            )
        return gdf

    def _slice_data(
        gdf: gpd.GeoDataFrame,
        variables: Optional[Union[str, List[str]]] = None,
        geom: Optional[Geom] = None,
        bbox: Optional[Geom] = None,
        crs: Optional[CRS] = None,
        buffer: float = 0.0,
        predicate: str = "intersects",  # TODO: enum available predicates
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,  # TODO: review NoDataStrategy + axes
        logger: Logger = logger,
    ) -> gpd.GeoDataFrame:
        """Return a clipped GeoDataFrame (vector).

        Arguments
        ---------
        gdf: gpd.GeoDataFrame
            GeoDataFrame to slice.
        variables : str or list of str, optional.
            Names of GeoDataFrame columns to return.
        geom : geopandas.GeoDataFrame/Series, optional
            A geometry defining the area of interest.
        bbox : array-like of floats, optional
            (xmin, ymin, xmax, ymax) bounding box of area of interest
            (in WGS84 coordinates).
        crs: pyproj.CRS
            Coordinate reference system of the bbox or geom.
        buffer : float, optional
            Buffer around the `bbox` or `geom` area of interest in meters. By default 0.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle no data values. By default NoDataStrategy.RAISE.
        predicate : str, optional
            Predicate used to filter the GeoDataFrame, see
            :py:func:`hydromt.gis_utils.filter_gdf` for details.
        handle_nodata: NoDataStrategy
            Strategy to use when resulting GeoDataFrame has no data.
        logger: Logging.Logger
            Python logger to use.

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame
        """
        if variables is not None:
            variables = np.atleast_1d(variables).tolist()
            if np.any([var not in gdf.columns for var in variables]):
                raise ValueError(f"GeoDataFrame: Not all variables found: {variables}")
            if "geometry" not in variables:  # always keep geometry column
                variables = variables + ["geometry"]
            gdf = gdf.loc[:, variables]

        if geom is not None or bbox is not None:
            # NOTE if we read with vector driver this is already done ..
            geom = parse_geom_bbox_buffer(geom, bbox, buffer, crs)
            bbox_str = ", ".join([f"{c:.3f}" for c in geom.total_bounds])
            epsg = geom.crs.to_epsg()
            logger.debug(f"Clip {predicate} [{bbox_str}] (EPSG:{epsg})")
            idxs = utils.filter_gdf(gdf, geom=geom, predicate=predicate)
            gdf = gdf.iloc[idxs]
        return gdf

    def _set_nodata(self, gdf: gpd.GeoDataFrame):
        # parse nodata values
        cols = gdf.select_dtypes([np.number]).columns
        if nodata := self.source.nodata is not None and len(cols) > 0:
            if not isinstance(nodata, dict):
                nodata = {c: nodata for c in cols}
            else:
                nodata = nodata
            for c in cols:
                mv = nodata.get(c, None)
                if mv is not None:
                    is_nodata = np.isin(gdf[c], np.atleast_1d(mv))
                    gdf[c] = np.where(is_nodata, np.nan, gdf[c])
        return gdf

    def _apply_unit_conversions(self, gdf: gpd.GeoDataFrame, logger: Logger = logger):
        # unit conversion
        unit_names = list(self.source.unit_mult.keys()) + list(
            self.source.unit_add.keys()
        )
        unit_names = [k for k in unit_names if k in gdf.columns]
        if len(unit_names) > 0:
            logger.debug(f"Convert units for {len(unit_names)} columns.")
        for name in list(set(unit_names)):  # unique
            m = self.source.unit_mult.get(name, 1)
            a = self.source.unit_add.get(name, 0)
            gdf[name] = gdf[name] * m + a
        return gdf

    def _set_metadata(self, gdf: gpd.GeoDataFrame):
        # set meta data
        gdf.attrs.update(self.source.meta)

        # set column attributes
        for col in self.source.attrs:
            if col in gdf.columns:
                gdf[col].attrs.update(**self.source.attrs[col])

        return gdf

    def get_bbox(self, detect: bool = True) -> TotalBounds:
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
        bbox = self.source.extent.get("bbox", None)
        crs = self.source.crs
        if bbox is None and detect:
            bbox, crs = self.detect_bbox()

        return bbox, crs

    # TODO: this should be done by the driver
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
            gdf = self.get_data()

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
            bbox, crs = self.get_bbox(detect=True)
            bbox = list(bbox)
            props = {**self.source.meta, "crs": crs}
            ext = splitext(self.source.uri)[-1]
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
                props = self.source.meta
                media_type = MediaType.JSON
            else:
                raise e
        else:
            stac_catalog = StacCatalog(
                self.source.name,
                description=self.source.name,
            )
            stac_item = StacItem(
                self.source.name,
                geometry=None,
                bbox=list(bbox),
                properties=props,
                datetime=datetime(1, 1, 1),
            )
            stac_asset = StacAsset(str(self.source.uri), media_type=media_type)
            base_name = basename(self.source.uri)
            stac_item.add_asset(base_name, stac_asset)

            stac_catalog.add_item(stac_item)
            return stac_catalog
