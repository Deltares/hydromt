"""Implementation for the geodataset DataAdapter."""

from logging import getLogger
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
    exec_nodata_strat,
)
from hydromt._typing.type_def import Number
from hydromt._utils import (
    _has_no_data,
    _rename_vars,
    _set_metadata,
    _set_vector_nodata,
    _shift_dataset_time,
    _single_var_as_array,
    _slice_temporal_dimension,
)
from hydromt.data_catalog.adapters.data_adapter_base import DataAdapterBase
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
    ) -> Optional[Union[xr.Dataset, xr.DataArray]]:
        """Return a clipped, sliced and harmonized RasterDataset.

        Parameters
        ----------
        ds : xr.Dataset
            input GeoDataset
        metadata : SourceMetadata
            source metadata
        mask : Optional[gpd.GeoDataFrame], optional
            mask to filter by geometry, by default None
        predicate : str, optional
            predicate to use for the mask filter, by default "intersects"
        variables : Optional[List[str]], optional
            variable filter, by default None
        time_range : Optional[TimeRange], optional
            filter start and end times, by default None
        single_var_as_array : bool, optional
            whether to return a xr.DataArray if only a single variable is present, by default True
        handle_nodata : NoDataStrategy, optional
            how to handle no data being present in the result, by default NoDataStrategy.RAISE

        Returns
        -------
        Optional[Union[xr.Dataset, xr.DataArray]]
            The filtered and harmonized GeoDataset, or None if no data was available

        Raises
        ------
        ValueError
            if not all variables are found in the data
        NoDataException
            if no data in left after slicing and handle_nodata is NoDataStrategy.RAISE
        """
        ds = _rename_vars(ds, self.rename)
        ds = GeoDatasetAdapter._validate_spatial_coords(ds)
        ds = GeoDatasetAdapter._set_crs(ds, crs=metadata.crs)
        ds = _set_vector_nodata(ds, metadata)
        ds = _shift_dataset_time(dt=self.unit_add.get("time", 0), ds=ds)
        ds = GeoDatasetAdapter._apply_unit_conversion(
            ds,
            unit_mult=self.unit_mult,
            unit_add=self.unit_add,
        )
        ds = _set_metadata(ds, metadata=metadata)
        ds = GeoDatasetAdapter._slice_data(
            ds,
            variables=variables,
            mask=mask,
            predicate=predicate,
            time_range=time_range,
        )

        if _has_no_data(ds):
            exec_nodata_strat(
                "No data was read from source",
                strategy=handle_nodata,
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
    ) -> Optional[xr.Dataset]:
        """Filter the GeoDataset.

        Parameters
        ----------
        ds : Optional[Union[xr.Dataset, xr.DataArray]]
            input dataset
        variables : Optional[List[str]], optional
            variable filter, by default None
        mask : Optional[gpd.GeoDataFrame], optional
            mask to filter by geometry, by default None
        predicate : str, optional
            predicate to use for the mask filter, by default "intersects"
        time_range : Optional[TimeRange], optional
            filter start and end times, by default None

        Returns
        -------
        Optional[xr.Dataset]
            the filtered GeoDataSet

        Raises
        ------
        ValueError
            if not all variables are found in the data
        NoDataException
            if no data in left after slicing and handle_nodata is NoDataStrategy.RAISE

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
            maybe_ds = _slice_temporal_dimension(ds, time_range)
        if mask is not None:
            maybe_ds = GeoDatasetAdapter._slice_spatial_dimension(
                maybe_ds,
                mask=mask,
                predicate=predicate,
            )
        if _has_no_data(maybe_ds):
            return None
        else:
            return maybe_ds

    @staticmethod
    def _slice_spatial_dimension(
        ds: Optional[xr.Dataset],
        mask: Geom,
        predicate: Predicate,
    ) -> Optional[xr.Dataset]:
        if ds is None:
            return None
        else:
            bbox_str = ", ".join([f"{c:.3f}" for c in mask.total_bounds])
            epsg = mask.crs.to_epsg()
            logger.debug(f"Clip {predicate} [{bbox_str}] (EPSG:{epsg})")
            ds = ds.vector.clip_geom(mask, predicate=predicate)
            if _has_no_data(ds):
                return None
            else:
                return ds

    @staticmethod
    def _apply_unit_conversion(
        ds: Optional[xr.Dataset],
        unit_mult: Dict[str, Number],
        unit_add: Dict[str, Number],
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
