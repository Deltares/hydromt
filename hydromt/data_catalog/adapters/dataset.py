"""Implementation for the RasterDatasetAdapter."""

from __future__ import annotations

from logging import getLogger
from typing import Optional

import numpy as np
import xarray as xr

from hydromt._typing import (
    Data,
    NoDataException,
    NoDataStrategy,
    SourceMetadata,
    TimeRange,
    Variables,
    exec_nodata_strat,
)
from hydromt._utils import (
    _has_no_data,
    _set_metadata,
    _shift_dataset_time,
    _single_var_as_array,
)
from hydromt.data_catalog.adapters.data_adapter_base import DataAdapterBase

logger = getLogger(__name__)

__all__ = ["DatasetAdapter"]


class DatasetAdapter(DataAdapterBase):
    """Implementation for the DatasetAdapter."""

    def transform(
        self,
        ds: xr.Dataset,
        metadata: SourceMetadata,
        *,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        single_var_as_array: bool = True,
    ) -> Optional[xr.Dataset]:
        """Return a clipped, sliced and unified Dataset.

        For a detailed description see:
        :py:func:`~hydromt.data_catalog.DataCatalog.get_dataset`
        """
        try:
            if ds is None:
                raise NoDataException()
            # rename variables and parse data and attrs
            ds = self._rename_vars(ds)
            ds = self._set_nodata(ds, metadata)
            ds = self._shift_time(ds)
            # slice
            ds = DatasetAdapter._slice_data(ds, variables, time_range)
            if ds is None:
                raise NoDataException()
            # uniformize
            ds = self._apply_unit_conversion(ds)
            ds = _set_metadata(ds, metadata)
            # return array if single var and single_var_as_array
            ds = _single_var_as_array(ds, single_var_as_array, variable_name=variables)
            return ds
        except NoDataException:
            exec_nodata_strat("No data to export", strategy=handle_nodata)

    def _rename_vars(self, ds: Data) -> Data:
        rm = {k: v for k, v in self.rename.items() if k in ds}
        ds = ds.rename(rm)
        return ds

    def _set_nodata(self, ds: Data, metadata: SourceMetadata) -> Data:
        if metadata.nodata is not None:
            if not isinstance(metadata.nodata, dict):
                nodata = {k: metadata.nodata for k in ds.data_vars.keys()}
            else:
                nodata = metadata.nodata
            for k in ds.data_vars:
                mv = nodata.get(k, None)
                if mv is not None and ds[k].attrs.get("_FillValue", None) is None:
                    ds[k].attrs["_FillValue"] = mv
        return ds

    def _apply_unit_conversion(self, ds: Data) -> Data:
        unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
        unit_names = [k for k in unit_names if k in ds.data_vars]
        if len(unit_names) > 0:
            logger.debug(f"Convert units for {len(unit_names)} variables.")
        for name in list(set(unit_names)):  # unique
            m = self.unit_mult.get(name, 1)
            a = self.unit_add.get(name, 0)
            da = ds[name]
            attrs = da.attrs.copy()
            nodata_isnan = da.attrs.get("_FillValue", None) is None or np.isnan(
                da.attrs.get("_FillValue", None)
            )
            # nodata value is explicitly set to NaN in case no nodata value is provided
            nodata = np.nan if nodata_isnan else da.attrs["_FillValue"]
            data_bool = ~np.isnan(da) if nodata_isnan else da != nodata
            ds[name] = xr.where(data_bool, da * m + a, nodata)
            ds[name].attrs.update(attrs)  # set original attributes
        return ds

    def _shift_time(self, ds: Data) -> Data:
        dt = self.unit_add.get("time", 0)
        return _shift_dataset_time(dt=dt, ds=ds)

    @staticmethod
    def _slice_data(
        ds: Data,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
    ) -> Optional[Data]:
        """Slice the dataset in space and time.

        Arguments
        ---------
        ds : xarray.Dataset or xarray.DataArray
            The Dataset to slice.
        variables : str or list of str, optional.
            Names of variables to return.
        time_range: tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.

        Returns
        -------
        ds : xarray.Dataset
            The sliced Dataset.
        """
        if isinstance(ds, xr.DataArray):
            if ds.name is None:
                ds.name = "data"
            ds = ds.to_dataset()
        elif variables is not None:
            variables = np.atleast_1d(variables).tolist()
            if len(variables) > 1 or len(ds.data_vars) > 1:
                mvars = [var not in ds.data_vars for var in variables]
                if any(mvars):
                    raise ValueError(f"Dataset: variables not found {mvars}")
                ds = ds[variables]
        if time_range is not None:
            ds = DatasetAdapter._slice_temporal_dimension(ds, time_range)

        if _has_no_data(ds):
            return None
        else:
            return ds

    @staticmethod
    def _slice_temporal_dimension(
        ds: Data,
        time_range: TimeRange,
    ) -> Optional[Data]:
        if (
            "time" in ds.dims
            and ds["time"].size > 1
            and np.issubdtype(ds["time"].dtype, np.datetime64)
        ):
            logger.debug(f"Slicing time dim {time_range}")
            ds = ds.sel(time=slice(*time_range))
        if _has_no_data(ds):
            return None
        else:
            return ds
