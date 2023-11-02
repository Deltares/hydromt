"""Implementation for the dataset DataAdapter."""
import logging
import os
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import List, NewType, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from hydromt.typing import ErrorHandleMethod

from .. import io
from .data_adapter import DataAdapter

logger = logging.getLogger(__name__)

DatasetSource = NewType("DatasetSource", str | Path)


class DatasetAdapter(DataAdapter):
    """DatasetAdapter for non-spatial n-dimensional data."""

    _DEFAULT_DRIVER = ""
    _DRIVERS = {
        "nc": "netcdf",
    }

    def __init__(
        self,
        path: str | Path,
        driver: Optional[str] = None,
        filesystem: Optional[str] = None,
        nodata: Optional[dict | float | int] = None,
        rename: Optional[dict] = None,
        unit_mult: Optional[dict] = None,
        unit_add: Optional[dict] = None,
        meta: Optional[dict] = None,
        attrs: Optional[dict] = None,
        driver_kwargs: Optional[dict] = None,
        storage_options: Optional[dict] = None,
        name: Optional[str] = "",
        catalog_name: Optional[str] = "",
        provider: Optional[str] = None,
        version: Optional[str] = None,
    ):
        """Docstring."""
        super().__init__(
            path=path,
            driver=driver,
            filesystem=filesystem,
            nodata=nodata,
            rename=rename,
            unit_mult=unit_mult,
            unit_add=unit_add,
            meta=meta,
            attrs=attrs,
            driver_kwargs=driver_kwargs,
            storage_options=storage_options,
            name=name,
            catalog_name=catalog_name,
            provider=provider,
            version=version,
        )

    def to_file(
        self,
        data_root: str | Path,
        data_name: str,
        time_tuple: Optional[Tuple[str | datetime]] = None,
        variables: Optional[List[str]] = None,
        driver: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        """Docstring."""
        obj = self.get_data(
            time_tuple=time_tuple,
            variables=variables,
            logger=logger,
            single_var_as_array=variables is None,
        )
        if driver is None or driver == "netcdf":
            # always write netcdf
            driver = "netcdf"
            dvars = [obj.name] if isinstance(obj, xr.DataArray) else obj.data_vars
            if variables is None:
                encoding = {k: {"zlib": True} for k in dvars}
                fn_out = join(data_root, f"{data_name}.nc")
                obj.to_netcdf(fn_out, encoding=encoding)
            else:  # save per variable
                if not os.path.isdir(join(data_root, data_name)):
                    os.makedirs(join(data_root, data_name))
                for var in dvars:
                    fn_out = join(data_root, data_name, f"{var}.nc")
                    obj[var].to_netcdf(fn_out, encoding={var: {"zlib": True}})
                fn_out = join(data_root, data_name, "{variable}.nc")
        elif driver == "zarr":
            fn_out = join(data_root, f"{data_name}.zarr")
            if isinstance(obj, xr.DataArray):
                obj = obj.to_dataset()
            obj.to_zarr(fn_out, **kwargs)
        else:
            raise ValueError(f"Dataset: Driver {driver} unknown.")

        return fn_out, driver

    def get_data(
        self,
        variables: Optional[List[str]] = None,
        time_tuple: Optional[Tuple[str | datetime]] = None,
        single_var_as_array: Optional[bool] = True,
        logger: Optional[logging.Logger] = logger,
    ) -> xr.Dataset:
        """Docstring."""
        # load data
        fns = self._resolve_paths(variables, time_tuple)
        ds = self._read_data(fns, logger=logger)
        self.mark_as_used()  # mark used
        # rename variables and parse data and attrs
        ds = self._rename_vars(ds)
        ds = self._set_nodata(ds)
        ds = self._shift_time(ds, logger=logger)
        # slice
        ds = DatasetAdapter._slice_data(ds, variables, time_tuple, logger=logger)
        # uniformize
        ds = self._apply_unit_conversion(ds, logger=logger)
        ds = self._set_metadata(ds)
        # return array if single var and single_var_as_array
        return self._single_var_as_array(ds, single_var_as_array, variables)

    def to_stac_catalog(self, on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE):
        """Docstring."""
        raise NotImplementedError

    def _read_data(self, fns, logger=logger):
        kwargs = self.driver_kwargs.copy()
        if len(fns) > 1 and self.driver in ["zarr"]:
            raise ValueError(
                f"Dataset: Reading multiple {self.driver} files is not supported."
            )
        logger.info(f"Reading {self.name} {self.driver} data from {self.path}")
        if self.driver in ["netcdf"]:
            ds = xr.open_mfdataset(fns, **kwargs)
        elif self.driver == "zarr":
            ds = xr.open_zarr(fns[0], **kwargs)
        elif self.driver == "mfcsv":
            fn_dict = {fn.stem: fn for fn in fns}
            ds = io.open_mfcsv(fn_dict, **kwargs)
        else:
            raise ValueError(f"Dataset: Driver {self.driver} unknown")

        return ds

    def _rename_vars(self, ds: xr.Dataset) -> xr.Dataset:
        rm = {k: v for k, v in self.rename.items() if k in ds}
        ds = ds.rename(rm)
        return ds

    def _set_metadata(self, ds: xr.DataArray | xr.Dataset) -> xr.Dataset | xr.DataArray:
        if self.attrs:
            if isinstance(ds, xr.DataArray):
                ds.attrs.update(self.attrs[ds.name])
            else:
                for k in self.attrs:
                    ds[k].attrs.update(self.attrs[k])

        ds.attrs.update(self.meta)
        return ds

    def _set_nodata(self, ds: xr.Dataset) -> xr.Dataset:
        if self.nodata is not None:
            if not isinstance(self.nodata, dict):
                nodata = {k: self.nodata for k in ds.data_vars.keys()}
            else:
                nodata = self.nodata
            for k in ds.data_vars:
                mv = nodata.get(k, None)
                if mv is not None and ds[k].attrs.get("_FillValue", None) is None:
                    ds[k].attrs["_FillValue"] = mv
        return ds

    def _apply_unit_conversion(
        self, ds: xr.Dataset, logger: logging.Logger = logger
    ) -> xr.Dataset:
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

    def _shift_time(
        self, ds: xr.Dataset, logger: logging.Logger = logger
    ) -> xr.Dataset:
        dt = self.unit_add.get("time", 0)
        if (
            dt != 0
            and "time" in ds.dims
            and ds["time"].size > 1
            and np.issubdtype(ds["time"].dtype, np.datetime64)
        ):
            logger.debug(f"Shifting time labels with {dt} sec.")
            ds["time"] = ds["time"] + pd.to_timedelta(dt, unit="s")
        elif dt != 0:
            logger.warning("Time shift not applied, time dimension not found.")
        return ds

    @staticmethod
    def _slice_data(
        ds: xr.Dataset,
        variables: Optional[str | List[str]] = None,
        time_tuple=Optional[Tuple[str] | Tuple[datetime]],
        logger: logging.Logger = logger,
    ) -> xr.Dataset:
        """Slice the dataset in space and time.

        Arguments
        ---------
        ds : xarray.Dataset or xarray.DataArray
            The Dataset to slice.
        variables : str or list of str, optional.
            Names of variables to return.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.

        Returns
        -------
        ds : xarray.Dataset
            The sliced Dataset.
        """
        if isinstance(ds, xr.DataArray):
            if ds.name is None:
                # dummy name, required to create dataset
                # renamed to variable in _single_var_as_array
                ds.name = "data"
            ds = ds.to_dataset()
        elif variables is not None:
            variables = np.atleast_1d(variables).tolist()
            if len(variables) > 1 or len(ds.data_vars) > 1:
                mvars = [var not in ds.data_vars for var in variables]
                if any(mvars):
                    raise ValueError(f"Dataset: variables not found {mvars}")
                ds = ds[variables]
        if time_tuple is not None:
            ds = DatasetAdapter._slice_temporal_dimension(ds, time_tuple, logger=logger)
        return ds

    @staticmethod
    def _slice_temporal_dimension(
        ds: xr.Dataset, time_tuple: tuple, logger: logging.Logger = logger
    ) -> xr.Dataset:
        if (
            "time" in ds.dims
            and ds["time"].size > 1
            and np.issubdtype(ds["time"].dtype, np.datetime64)
        ):
            logger.debug(f"Slicing time dim {time_tuple}")
            ds = ds.sel(time=slice(*time_tuple))
            if ds.time.size == 0:
                raise IndexError("Dataset: Time slice out of range.")
        return ds
