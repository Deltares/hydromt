"""Implementation for the dataset DataAdapter."""
import logging
from datetime import datetime
from os.path import basename, splitext
from pathlib import Path
from typing import List, NewType, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from pystac import MediaType

from hydromt.typing import ErrorHandleMethod, TimeRange

from .data_adapter import DataAdapter
from .utils import netcdf_writer, shift_dataset_time, zarr_writer

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
        """Initiate data adapter for n-dimensional timeseries data.

        This object contains all properties required to read supported files(netcdf, zarr) into
        a single unified Dataset, i.e. :py:class:`xarray.Dataset`. In addition it keeps meta data to be able to reproduce which
        data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source. If the dataset consists of multiple files, the path may
            contain {variable}, {year}, {month} placeholders as well as path
            search pattern using a ``*`` wildcard.
        driver: {'netcdf', 'zarr'}, optional
            Driver to read files with,
            for 'netcdf' :py:func:`xarray.open_mfdataset`.
            By default the driver is inferred from the file extension.
        filesystem: str, optional
            Filesystem where the data is stored (local, cloud, http etc.).
            If None (default) the filesystem is inferred from the path.
            See :py:func:`fsspec.registry.known_implementations` for all options.
        nodata: float, int, optional
            Missing value number. Only used if the data has no native missing value.
            Nodata values can be differentiated between variables using a dictionary.
        rename: dict, optional
            Mapping of native data source variable to output source variable name as
            required by hydroMT.
        unit_mult, unit_add: dict, optional
            Scaling multiplication and addition to change to map from the native
            data unit to the output data unit as required by hydroMT.
        meta: dict, optional
            Metadata information of dataset, prefably containing the following keys:
            - 'source_version'
            - 'source_url'
            - 'source_license'
            - 'paper_ref'
            - 'paper_doi'
            - 'category'
        placeholders: dict, optional
            Placeholders to expand yaml entry to multiple entries (name and path)
            based on placeholder values
        attrs: dict, optional
            Additional attributes relating to data variables. For instance unit
            or long name of the variable.
        driver_kwargs, dict, optional
            Additional key-word arguments passed to the driver.
        storage_options: dict, optional
            Additional key-word arguments passed to the fsspec FileSystem object.
        name, catalog_name: str, optional
            Name of the dataset and catalog, optional.
        provider: str, optional
            A name to identifiy the specific provider of the dataset requested.
            if None is provided, the last added source will be used.
        version: str, optional
            A name to identifiy the specific version of the dataset requested.
            if None is provided, the last added source will be used.
        """
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
        time_tuple: Optional[Tuple[str, str] | Tuple[datetime, datetime]] = None,
        variables: Optional[List[str]] = None,
        driver: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        """Save a dataset slice to file. By default the data is saved as a NetCDF file.

        Parameters
        ----------
        data_root : str, Path
            Path to output folder
        data_name : str
            Name of output file without extension.
        variables : list of str, optional
            Names of Dataset variables to return. By default all dataset variables
            are returned.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        driver : str, optional
            Driver to write file, e.g.: 'netcdf', 'zarr', by default None
        **kwargs
            Additional keyword arguments that are passed to the `to_zarr`
            function.

        Returns
        -------
        fn_out: str
            Absolute path to output file
        driver: str
            Name of driver to read data with, see
            :py:func:`~hydromt.data_catalog.DataCatalog.get_dataset`
        """
        obj = self.get_data(
            time_tuple=time_tuple,
            variables=variables,
            logger=logger,
            single_var_as_array=variables is None,
        )
        if driver is None or driver == "netcdf":
            fn_out = netcdf_writer(
                obj=obj, data_root=data_root, data_name=data_name, variables=variables
            )
        elif driver == "zarr":
            fn_out = zarr_writer(
                obj=obj, data_root=data_root, data_name=data_name, **kwargs
            )
        else:
            raise ValueError(f"Dataset: Driver {driver} unknown.")

        return fn_out, driver

    def get_data(
        self,
        variables: Optional[List[str]] = None,
        time_tuple: Optional[Tuple[str, str] | Tuple[datetime, datetime]] = None,
        single_var_as_array: Optional[bool] = True,
        logger: Optional[logging.Logger] = logger,
    ) -> xr.Dataset:
        """Return a clipped, sliced and unified Dataset.

        For a detailed description see:
        :py:func:`~hydromt.data_catalog.DataCatalog.get_dataset`
        """
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
            raise NotImplementedError
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
        return shift_dataset_time(dt=dt, ds=ds, logger=logger)

    @staticmethod
    def _slice_data(
        ds: xr.Dataset,
        variables: Optional[str | List[str]] = None,
        time_tuple=Optional[Tuple[str] | Tuple[datetime]],
        logger: logging.Logger = logger,
        on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE,
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
            try:
                ds = DatasetAdapter._slice_temporal_dimension(
                    ds, time_tuple, logger=logger
                )
            except IndexError as e:
                if on_error == ErrorHandleMethod.SKIP:
                    logger.warning(
                        "Skipping slicing data on temporal dimension because time slice is out of range."
                    )
                    return ds
                elif on_error == ErrorHandleMethod.COERCE:
                    return ds
                else:
                    raise e

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

    def get_time_range(
        self, ds: Optional[xr.DataArray | xr.Dataset] = None
    ) -> TimeRange:
        """Get the temporal range of a dataset.

        Parameters
        ----------
        ds : Optional[xr.DataArray  |  xr.Dataset]
            The dataset to detect the time range of. It must have a time dimentsion set.
            If none is provided, :py:meth:`hydromt.DatasetAdapter.get_data`
            will be used to fetch the it before detecting.


        Returns
        -------
        range: Tuple[np.datetime64, np.datetime64]
            A tuple containing the start and end of the time dimension. Range is
            inclusive on both sides.
        """
        if ds is None:
            ds = self.get_data()

        try:
            return (ds.time[0].values, ds.time[-1].values)
        except AttributeError:
            raise AttributeError("Dataset has no dimension called 'time'")

    def to_stac_catalog(self, on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE):
        """
        Convert a dataset into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - on_error (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "skip" to skip the
          dataset on failure, and "coerce" (default) to set default values on failure.

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataset, or None
          if the dataset was skipped.
        """
        try:
            start_dt, end_dt = self.get_time_range()
            start_dt = pd.to_datetime(start_dt)
            end_dt = pd.to_datetime(end_dt)
            props = {**self.meta}
            ext = splitext(self.path)[-1]
            bbox = [0.0, 0.0, 0.0, 0.0]

            match ext:
                case ".nc":
                    media_type = MediaType.HDF5

                case ".zarr":
                    raise RuntimeError("STAC does not support zarr datasets")

                case _:
                    raise RuntimeError(
                        f"Unknown extention: {ext} cannot determine media type"
                    )
        except (IndexError, KeyError) as e:
            if on_error == ErrorHandleMethod.SKIP:
                logger.warning(
                    "Skipping {name} during stac conversion because"
                    "because detecting temporal extent failed."
                )
                return
            elif on_error == ErrorHandleMethod.COERCE:
                bbox = [0.0, 0.0, 0.0, 0.0]
                props = self.meta
                start_dt = datetime(1, 1, 1)
                end_dt = datetime(1, 1, 1)
                media_type = MediaType.JSON
            else:
                raise e
        stac_catalog = StacCatalog(self.name, description=self.name)
        stac_item = StacItem(
            self.name,
            geometry=None,
            bbox=bbox,
            properties=props,
            datetime=None,
            start_datetime=start_dt,
            end_datetime=end_dt,
        )
        stac_asset = StacAsset(str(self.path), media_type=media_type)
        base_name = basename(self.path)
        stac_item.add_asset(base_name, stac_asset)
        stac_catalog.add_item(stac_item)
        return stac_catalog
