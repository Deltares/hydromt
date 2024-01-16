"""The GeoDataFrame adapter performs transformations on GeoDataFrames."""
from logging import Logger, getLogger

import geopandas as gpd
import numpy as np
from pyproj import CRS

from hydromt.data_sources.geodataframe_data_source import GeoDataFrameDataSource
from hydromt.gis_utils import filter_gdf, parse_geom_bbox_buffer
from hydromt.nodata import NoDataStrategy, _exec_nodata_strat
from hydromt.typing import GEOM_TYPES

from .data_adapter_base import DataAdapterBase

logger: Logger = getLogger(__name__)


class GeoDataFrameAdapter(DataAdapterBase):
    """The GeoDataFrameAdapter performs transformations on GeoDataFrames."""

    source: GeoDataFrameDataSource

    def _slice_data(
        gdf: gpd.GeoDataFrame,
        variables: str | list[str] | None = None,
        geom: GEOM_TYPES | None = None,
        bbox: GEOM_TYPES | None = None,
        crs: CRS | None = None,
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
            geom = parse_geom_bbox_buffer(geom, bbox, buffer)
            bbox_str = ", ".join([f"{c:.3f}" for c in geom.total_bounds])
            epsg = geom.crs.to_epsg()
            logger.debug(f"Clip {predicate} [{bbox_str}] (EPSG:{epsg})")
            idxs = filter_gdf(gdf, geom=geom, predicate=predicate)
            if idxs.size == 0:
                _exec_nodata_strat(
                    "No data within spatial domain.", handle_nodata, logger=logger
                )
            gdf = gdf.iloc[idxs]
        return gdf
