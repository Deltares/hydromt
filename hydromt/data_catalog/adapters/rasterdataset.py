"""Implementation for the RasterDatasetAdapter."""

from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pyproj
import xarray as xr
from pyproj import CRS

from hydromt._typing import (
    Data,
    Geom,
    NoDataException,
    NoDataStrategy,
    SourceMetadata,
    TimeRange,
    Variables,
    exec_nodata_strat,
)
from hydromt._utils import (
    _has_no_data,
    _shift_dataset_time,
    _single_var_as_array,
    _slice_temporal_dimension,
)
from hydromt.data_catalog.adapters.data_adapter_base import DataAdapterBase
from hydromt.gis._raster_utils import _meridian_offset
from hydromt.gis.raster import GEO_MAP_COORD

logger = getLogger(__name__)

__all__ = ["RasterDatasetAdapter"]


class RasterDatasetAdapter(DataAdapterBase):
    """Implementation for the RasterDatasetAdapter."""

    def transform(
        self,
        ds: xr.Dataset,
        metadata: SourceMetadata,
        *,
        mask: Optional[Geom] = None,
        align: Optional[bool] = None,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        single_var_as_array: bool = True,
    ) -> Union[xr.Dataset, xr.DataArray, None]:
        """Filter and harmonize the input RasterDataset.

        Parameters
        ----------
        ds : xr.Dataset
            input RasterDataset
        metadata : SourceMetadata
            source metadata
        mask : Optional[gpd.GeoDataFrame], optional
            mask to filter by geometry, by default None
        variables : Optional[List[str]], optional
            variable filter, by default None
        time_range : Optional[TimeRange], optional
            filter start and end times, by default None
        handle_nodata : NoDataStrategy, optional
            how to handle no data being present in the result, by default NoDataStrategy.RAISE
        single_var_as_array : bool, optional
            whether to return a xr.DataArray if only a single variable is present, by default True

        Returns
        -------
        Optional[xr.Dataset]
            The filtered and harmonized RasterDataset, or None if no data was available

        Raises
        ------
        ValueError
            if not all variables are found in the data
        NoDataException
            if no data in left after slicing and handle_nodata is NoDataStrategy.RAISE
        """
        try:
            if _has_no_data(ds):
                raise NoDataException()
            # rename variables and parse data and attrs
            ds = self._rename_vars(ds)
            ds = self._validate_spatial_dims(ds)
            ds = self._set_crs(ds, metadata.crs)
            ds = self._set_nodata(ds, metadata)
            ds = _shift_dataset_time(dt=self.unit_add.get("time", 0), ds=ds)
            # slice data
            ds = RasterDatasetAdapter._slice_data(
                ds,
                variables,
                mask,
                align,
                time_range,
            )
            if _has_no_data(ds):
                raise NoDataException()

            # uniformize data
            ds = self._apply_unit_conversions(ds)
            ds = self._set_metadata(ds, metadata)
            # return array if single var and single_var_as_array
            return _single_var_as_array(ds, single_var_as_array, variables)
        except NoDataException:
            exec_nodata_strat(
                "No data was read from source",
                strategy=handle_nodata,
            )

    def _rename_vars(self, ds: Data) -> Data:
        rm = {k: v for k, v in self.rename.items() if k in ds}
        ds = ds.rename(rm)
        return ds

    def _validate_spatial_dims(self, ds: Data) -> Data:
        if GEO_MAP_COORD in ds.data_vars:
            ds = ds.set_coords(GEO_MAP_COORD)
        try:
            ds.raster.set_spatial_dims()
            # transpose dims to get y and x dim last
            x_dim = ds.raster.x_dim
            y_dim = ds.raster.y_dim
            ds = ds.transpose(..., y_dim, x_dim)
        except ValueError:
            raise ValueError(
                f"RasterDataset: No valid spatial coords found in data {self.path}"
            )
        return ds

    def _set_crs(self, ds: Data, crs: Optional[CRS]) -> Data:
        # set crs
        if ds.raster.crs is None and crs is not None:
            ds.raster.set_crs(crs)
        elif ds.raster.crs is None:
            raise ValueError("RasterDataset: CRS not defined in data catalog or data.")
        elif crs is not None and ds.raster.crs != pyproj.CRS.from_user_input(crs):
            logger.warning(
                "RasterDataset: CRS from data catalog does not match CRS "
                " of data. The original CRS will be used. Please check your catalog."
            )
        return ds

    @staticmethod
    def _slice_data(
        ds: Data,
        variables: Optional[List] = None,
        mask: Optional[Geom] = None,
        align: Optional[float] = None,
        time_range: Optional[TimeRange] = None,
    ) -> Optional[xr.Dataset]:
        """Filter the RasterDataset.

        Parameters
        ----------
        ds : xr.Dataset
            input Dataset.
        variables : Optional[List[str]], optional
            variable filter, by default None
        mask : Optional[gpd.GeoDataFrame], optional
            mask to filter by geometry, by default None
        align : Optional[float], optional
            resolution to align the bounding box, by default None
        time_range : Optional[TimeRange], optional
            filter start and end times, by default None

        Returns
        -------
        Optional[Union[xr.Dataset, xr.DataArray]]
            The filtered and harmonized RasterDataset, or None if no data was available

        Raises
        ------
        ValueError
            if not all variables are found in the data
        NoDataException
            if no data in left after slicing and handle_nodata is NoDataStrategy.RAISE
        """
        if isinstance(ds, xr.DataArray):  # xr.DataArray has no variables
            if ds.name is None:
                # dummy name, required to create dataset
                # renamed to variable in _single_var_as_array
                ds.name = "data"
            ds = ds.to_dataset()
        elif variables is not None:  # xr.Dataset has variables
            variables = cast(List, np.atleast_1d(variables).tolist())
            if len(variables) > 1 or len(ds.data_vars) > 1:
                mvars = [var not in ds.data_vars for var in variables]
                if any(mvars):
                    raise NoDataException(f"RasterDataset: variables not found {mvars}")
                ds = ds[variables]
        if time_range is not None:
            ds = _slice_temporal_dimension(ds, time_range)
        if mask is not None:
            ds = RasterDatasetAdapter._slice_spatial_dimensions(ds, mask, align)

        if _has_no_data(ds):
            return None
        else:
            return ds

    @staticmethod
    def _slice_spatial_dimensions(
        ds: Data,
        mask: Optional[Geom] = None,
        align: Optional[float] = None,
    ):
        # make sure bbox is in data crs
        bbox = None
        crs = ds.raster.crs
        epsg = crs.to_epsg()  # this could return None
        if mask is not None:
            bbox = mask.to_crs(crs).total_bounds
        # work with 4326 data that is defined at 0-360 degrees longtitude
        w, _, e, _ = ds.raster.bounds
        if epsg == 4326 and np.isclose(e - w, 360):  # allow for rounding errors
            ds = _meridian_offset(ds, bbox)

        # clip with bbox
        if bbox is not None:
            # check if bbox is fully covered
            bbox_str = ", ".join([f"{c:.3f}" for c in bbox])

            def _lt_or_close(a: float, b: float) -> bool:
                return np.isclose(a, b) or a < b

            w, s, e, n = ds.raster.bounds

            if not any(
                map(_lt_or_close, (w, s, bbox[2], bbox[3]), (bbox[0], bbox[1], e, n))
            ):
                logger.warning(
                    f"Dataset [{w}, {s}, {e}, {n}] does not fully cover bbox [{bbox_str}]"
                )

            logger.debug(f"Clip to [{bbox_str}] (epsg:{epsg}))")
            ds = ds.raster.clip_bbox(bbox, align=align)

        if _has_no_data(ds):
            return None
        else:
            return ds

    def _apply_unit_conversions(self, ds: Data):
        unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
        unit_names = [k for k in unit_names if k in ds.data_vars]
        if len(unit_names) > 0:
            logger.debug(f"Convert units for {len(unit_names)} variables.")
        for name in list(set(unit_names)):  # unique
            m = self.unit_mult.get(name, 1)
            a = self.unit_add.get(name, 0)
            da = ds[name]
            attrs = da.attrs.copy()
            nodata_isnan = da.raster.nodata is None or np.isnan(da.raster.nodata)
            # nodata value is explicitly set to NaN in case no nodata value is provided
            nodata = np.nan if nodata_isnan else da.raster.nodata
            data_bool = ~np.isnan(da) if nodata_isnan else da != nodata
            ds[name] = xr.where(data_bool, da * m + a, nodata)
            ds[name].attrs.update(attrs)  # set original attributes
            ds[name].raster.set_nodata(nodata)  # reset nodata in case of change

        return ds

    def _set_nodata(self, ds, metadata: SourceMetadata):
        """Parse and apply nodata values from the data catalog."""
        # set nodata value
        if nodata := metadata.nodata is not None:
            if not isinstance(nodata, dict):
                no_data_values: Dict[str, Any] = {
                    k: metadata.nodata for k in ds.data_vars.keys()
                }
            else:
                no_data_values: Dict[str, Any] = nodata
            for k in ds.data_vars:
                mv = no_data_values.get(k, None)
                if mv is not None and ds[k].raster.nodata is None:
                    ds[k].raster.set_nodata(mv)
        return ds

    def _set_metadata(self, ds, metadata: SourceMetadata):
        # unit attributes
        attrs = metadata.attrs
        for k in attrs:
            ds[k].attrs.update(attrs[k])
        # set meta data
        ds.attrs.update(metadata.model_dump(exclude=["attrs"], exclude_unset=True))
        return ds
