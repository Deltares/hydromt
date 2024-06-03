"""The GeoDataFrame adapter performs transformations on GeoDataFrames."""

from logging import Logger, getLogger
from typing import Any, Dict, Iterable, List, Optional, Union

import geopandas as gpd
import numpy as np
import pyproj
from pyproj import CRS

from hydromt._typing import (
    Geom,
    NoDataStrategy,
    SourceMetadata,
)
from hydromt._typing.error import NoDataException, _exec_nodata_strat
from hydromt.gis import utils

from .data_adapter_base import DataAdapterBase

logger: Logger = getLogger(__name__)


class GeoDataFrameAdapter(DataAdapterBase):
    """The GeoDataFrameAdapter performs transformations on GeoDataFrames."""

    def transform(
        self,
        gdf: gpd.GeoDataFrame,
        metadata: "SourceMetadata",
        *,
        mask: Optional[gpd.GeoDataFrame] = None,
        variables: Optional[List[str]] = None,
        predicate: str = "intersects",
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
    ) -> Optional[gpd.GeoDataFrame]:
        """Read transform data to HydroMT standards."""
        # rename variables and parse crs & nodata
        try:
            gdf = self._rename_vars(gdf)
            gdf = self._set_crs(gdf, metadata.crs, logger=logger)
            gdf = self._set_nodata(gdf, metadata)
            # slice
            gdf: Optional[gpd.GeoDataFrame] = GeoDataFrameAdapter._slice_data(
                gdf,
                variables=variables,
                mask=mask,
                predicate=predicate,
                handle_nodata=handle_nodata,
                logger=logger,
            )
            # uniformize
            if gdf is not None:
                gdf = self._apply_unit_conversions(gdf, logger=logger)
                gdf = self._set_metadata(gdf, metadata)
            return gdf
        except NoDataException:
            _exec_nodata_strat(
                "No data was read from source",
                strategy=handle_nodata,
                logger=logger,
            )

    def _rename_vars(self, gdf: gpd.GeoDataFrame):
        # rename and select columns
        if rename := self.rename:
            rename = {k: v for k, v in rename.items() if k in gdf.columns}
            gdf = gdf.rename(columns=rename)
        return gdf

    def _set_crs(self, gdf: gpd.GeoDataFrame, crs: Optional[CRS], logger=logger):
        if crs is not None and gdf.crs is None:
            gdf.set_crs(crs, inplace=True)
        elif gdf.crs is None:
            raise ValueError("GeoDataFrame: CRS not defined in data catalog or data.")
        elif crs is not None and gdf.crs != pyproj.CRS.from_user_input(crs):
            logger.warning(
                "GeoDataFrame : CRS from data catalog does not match CRS of"
                " data. The original CRS will be used. Please check your data catalog."
            )
        return gdf

    @staticmethod
    def _slice_data(
        gdf: gpd.GeoDataFrame,
        variables: Optional[Union[str, List[str]]] = None,
        mask: Optional[Geom] = None,
        predicate: str = "intersects",  # TODO: enum available predicates
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,  # TODO: review NoDataStrategy + axes
        logger: Logger = logger,
    ) -> Optional[gpd.GeoDataFrame]:
        """Return a clipped GeoDataFrame (vector).

        Arguments
        ---------
        gdf: gpd.GeoDataFrame
            GeoDataFrame to slice.
        variables : str or list of str, optional.
            Names of GeoDataFrame columns to return.
        mask: geopandas.GeoDataFrame/Series, optional
            A geometry defining the area of interest.
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

        if mask is not None:
            # NOTE if we read with vector driver this is already done ..
            bbox_str = ", ".join([f"{c:.3f}" for c in mask.total_bounds])
            epsg = mask.crs.to_epsg()
            logger.debug(f"Clip {predicate} [{bbox_str}] (EPSG:{epsg})")
            idxs = utils.filter_gdf(gdf, geom=mask, predicate=predicate)
            gdf = gdf.iloc[idxs]

        if np.all(gdf.is_empty):
            _exec_nodata_strat(
                "No data was read from source", strategy=handle_nodata, logger=logger
            )
            gdf = None
        return gdf

    def _set_nodata(self, gdf: gpd.GeoDataFrame, metadata: "SourceMetadata"):
        """Parse and apply nodata values from the data catalog."""
        cols: Iterable[str] = gdf.select_dtypes([np.number]).columns
        no_data_value: Union[Dict[str, Any], str, None]
        if no_data_value := metadata.nodata is not None and len(cols) > 0:
            if not isinstance(no_data_value, dict):
                no_data_dict: Dict[str, Any] = {c: no_data_value for c in cols}
            else:
                no_data_dict: Dict[str, Any] = no_data_value
            for c in cols:
                mv = no_data_dict.get(c, None)
                if mv is not None:
                    is_nodata = np.isin(gdf[c], np.atleast_1d(mv))
                    gdf[c] = np.where(is_nodata, np.nan, gdf[c])
        return gdf

    def _apply_unit_conversions(self, gdf: gpd.GeoDataFrame, logger: Logger = logger):
        # unit conversion
        unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
        unit_names = [k for k in unit_names if k in gdf.columns]
        if len(unit_names) > 0:
            logger.debug(f"Convert units for {len(unit_names)} columns.")
        for name in list(set(unit_names)):  # unique
            m = self.unit_mult.get(name, 1)
            a = self.unit_add.get(name, 0)
            gdf[name] = gdf[name] * m + a
        return gdf

    def _set_metadata(self, gdf: gpd.GeoDataFrame, metadata: "SourceMetadata"):
        # set meta data
        gdf.attrs.update(metadata)

        # set column attributes
        for col in metadata.attrs:
            if col in gdf.columns:
                gdf[col].attrs.update(**metadata.attrs[col])

        return gdf
