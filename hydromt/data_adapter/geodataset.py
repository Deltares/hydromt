"""Implementation for the geodataset DataAdapter."""

from logging import Logger, getLogger
from typing import Dict, List, Optional, Union, cast

import numpy as np
import pyproj
import xarray as xr

from hydromt._typing import (
    Geom,
    NoDataStrategy,
    Predicate,
    SourceMetadata,
    TimeRange,
    Variables,
    _exec_nodata_strat,
)
from hydromt._typing.type_def import Number
from hydromt.data_adapter.data_adapter_base import DataAdapterBase
from hydromt.data_adapter.utils import (
    _rename_vars,
    _set_metadata,
    _set_vector_nodata,
    _single_var_as_array,
    _slice_temporal_dimension,
    has_no_data,
    shift_dataset_time,
)
from hydromt.gis.raster import GEO_MAP_COORD

logger = getLogger(__name__)

__all__ = ["GeoDatasetAdapter"]


class GeoDatasetAdapter(DataAdapterBase):
    """DatasetAdapter for GeoDatasets."""

    def transform(
        self,
        ds: xr.Dataset,
        metadata: SourceMetadata,
        *,
        mask: Optional[Geom] = None,
        predicate: Predicate = "intersects",
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        single_var_as_array: bool = True,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
    ) -> Optional[Union[xr.Dataset, xr.DataArray]]:
        """Return a clipped, sliced and unified RasterDataset.

        For a detailed description see:
        :py:func:`~hydromt.data_catalog.DataCatalog.get_rasterdataset`
        """
        ds = _rename_vars(ds, self.rename)
        ds = GeoDatasetAdapter._validate_spatial_coords(ds)
        ds = GeoDatasetAdapter._set_crs(ds, crs=metadata.crs, logger=logger)
        ds = _set_vector_nodata(ds, metadata)
        ds = shift_dataset_time(dt=self.unit_add.get("time", 0), ds=ds, logger=logger)
        ds = GeoDatasetAdapter._apply_unit_conversion(
            ds,
            unit_mult=self.unit_mult,
            unit_add=self.unit_add,
            logger=logger,
        )
        ds = _set_metadata(ds, metadata=metadata)
        ds = GeoDatasetAdapter._slice_data(
            ds,
            variables=variables,
            mask=mask,
            predicate=predicate,
            time_range=time_range,
            logger=logger,
        )

        if has_no_data(ds):
            _exec_nodata_strat(
                "No data was read from source",
                strategy=handle_nodata,
                logger=logger,
            )
            return None  # if handle_nodata ignore
        return _single_var_as_array(ds, single_var_as_array, variables)

    @staticmethod
    def _validate_spatial_coords(
        ds: Optional[xr.Dataset],
    ) -> Optional[xr.Dataset]:
        if ds is None:
            return None
        if GEO_MAP_COORD in ds.data_vars:
            ds = ds.set_coords(GEO_MAP_COORD)
        try:
            ds.vector.set_spatial_dims()
            idim = ds.vector.index_dim
            if idim not in ds:  # set coordinates for index dimension if missing
                ds[idim] = xr.IndexVariable(idim, np.arange(ds.dims[idim]))
            coords = [ds.vector.x_name, ds.vector.y_name, idim]
            coords = [item for item in coords if item is not None]
            ds = ds.set_coords(coords)
        except ValueError:
            raise ValueError("GeoDataset: No spatial geometry dimension found")
        return ds

    @staticmethod
    def _set_crs(
        ds: Optional[xr.Dataset],
        crs: Union[str, int, None] = None,
        logger: Logger = logger,
    ) -> Optional[xr.Dataset]:
        if ds is None:
            return None
        # set crs
        if ds.vector.crs is None and crs is not None:
            ds.vector.set_crs(crs)
        elif ds.vector.crs is None:
            raise ValueError("GeoDataset: CRS not defined in data catalog or data.")
        elif crs is not None and ds.vector.crs != pyproj.CRS.from_user_input(crs):
            logger.warning(
                "GeoDataset: CRS from data catalog does not match CRS of"
                " data. The original CRS will be used. Please check your data catalog."
            )
        return ds

    @staticmethod
    def _slice_data(
        ds: Optional[Union[xr.Dataset, xr.DataArray]],
        variables: Optional[Variables] = None,
        mask: Optional[Geom] = None,
        predicate: Predicate = "intersects",
        time_range: Optional[TimeRange] = None,
        logger: Logger = logger,
    ) -> Optional[xr.Dataset]:
        """Slice the dataset in space and time.

        Arguments
        ---------
        ds : xarray.Dataset or xarray.DataArray
            The GeoDataset to slice.
        variables : str or list of str, optional.
            Names of variables to return.
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest
            (in WGS84 coordinates).
        buffer : float, optional
            Buffer distance [m] applied to the geometry or bbox. By default 0 m.
        predicate : str, optional
            Predicate used to filter the GeoDataFrame, see
            :py:func:`hydromt.gis.utils.filter_gdf` for details.
        handle_nodata : NoDataStrategy, optional
            How to handle no data values. By default NoDataStrategy.RAISE.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.

        Returns
        -------
        ds : xarray.Dataset
            The sliced GeoDataset.
        """
        if isinstance(ds, xr.DataArray):
            if ds.name is None:
                # dummy name, required to create dataset
                # renamed to variable in _single_var_as_array
                ds.name = "data"
            ds = ds.to_dataset()
        elif variables is not None:
            assert isinstance(ds, xr.Dataset)
            variables = cast(List, np.atleast_1d(variables).tolist())
            if len(variables) > 1 or len(ds.data_vars) > 1:
                mvars = [var not in ds.data_vars for var in variables]
                if any(mvars):
                    raise ValueError(f"GeoDataset: variables not found {mvars}")
                ds = ds[variables]
        maybe_ds: Optional[xr.Dataset] = ds
        if time_range is not None:
            maybe_ds = _slice_temporal_dimension(ds, time_range, logger=logger)
        if mask is not None:
            maybe_ds = GeoDatasetAdapter._slice_spatial_dimension(
                maybe_ds,
                mask=mask,
                predicate=predicate,
                logger=logger,
            )
        if has_no_data(maybe_ds):
            return None
        else:
            return maybe_ds

    @staticmethod
    def _slice_spatial_dimension(
        ds: Optional[xr.Dataset],
        mask: Geom,
        predicate: Predicate,
        logger: Logger = logger,
    ) -> Optional[xr.Dataset]:
        if ds is None:
            return None
        else:
            bbox_str = ", ".join([f"{c:.3f}" for c in mask.total_bounds])
            epsg = mask.crs.to_epsg()
            logger.debug(f"Clip {predicate} [{bbox_str}] (EPSG:{epsg})")
            ds = ds.vector.clip_geom(mask, predicate=predicate)
            if has_no_data(ds):
                return None
            else:
                return ds

    @staticmethod
    def _apply_unit_conversion(
        ds: Optional[xr.Dataset],
        unit_mult: Dict[str, Number],
        unit_add: Dict[str, Number],
        logger: Logger = logger,
    ) -> Optional[xr.Dataset]:
        if ds is None:
            return None

        unit_names = list(unit_mult.keys()) + list(unit_add.keys())
        unit_names = [k for k in unit_names if k in ds.data_vars]
        if len(unit_names) > 0:
            logger.debug(f"Convert units for {len(unit_names)} variables.")
        for name in list(set(unit_names)):  # unique
            m = unit_mult.get(name, 1)
            a = unit_add.get(name, 0)
            da = ds[name]
            attrs = da.attrs.copy()
            nodata_isnan = da.vector.nodata is None or np.isnan(da.vector.nodata)
            # nodata value is explicitly set to NaN in case no nodata value is provided
            nodata = np.nan if nodata_isnan else da.vector.nodata
            data_bool = ~np.isnan(da) if nodata_isnan else da != nodata
            ds[name] = xr.where(data_bool, da * m + a, nodata)
            ds[name].attrs.update(attrs)  # set original attributes
        return ds
