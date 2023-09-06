"""Implementation for the geodataset DataAdapter."""
import logging
import os
import warnings
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import NewType, Tuple, Union

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from shapely import geometry
from shapely.geometry import box

from .. import gis_utils, io
from ..raster import GEO_MAP_COORD
from .data_adapter import DataAdapter

logger = logging.getLogger(__name__)

__all__ = ["GeoDatasetAdapter", "GeoDatasetSource"]

GeoDatasetSource = NewType("GeoDatasetSource", Union[str, Path])


class GeoDatasetAdapter(DataAdapter):

    """DatasetAdapter for GeoDatasets."""

    _DEFAULT_DRIVER = "vector"
    _DRIVERS = {
        "nc": "netcdf",
    }

    def __init__(
        self,
        path: str,
        driver: str = None,
        filesystem: str = "local",
        crs: Union[int, str, dict] = None,
        nodata: Union[dict, float, int] = None,
        rename: dict = {},
        unit_mult: dict = {},
        unit_add: dict = {},
        meta: dict = {},
        attrs: dict = {},
        driver_kwargs: dict = {},
        name: str = "",  # optional for now
        catalog_name: str = "",  # optional for now
        provider=None,
        version=None,
        **kwargs,
    ):
        """Initiate data adapter for geospatial timeseries data.

        This object contains all properties required to read supported files into
        a single unified GeoDataset, i.e. :py:class:`xarray.Dataset` with geospatial
        point geometries. In addition it keeps meta data to be able to reproduce which
        data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source. If the dataset consists of multiple files, the path may
            contain {variable}, {year}, {month} placeholders as well as path
            search pattern using a '*' wildcard.
        driver: {'vector', 'netcdf', 'zarr'}, optional
            Driver to read files with,
            for 'vector' :py:func:`~hydromt.io.open_geodataset`,
            for 'netcdf' :py:func:`xarray.open_mfdataset`.
            By default the driver is inferred from the file extension and falls back to
            'vector' if unknown.
        filesystem: {'local', 'gcs', 's3'}, optional
            Filesystem where the data is stored (local, cloud, http etc.).
            By default, local.
        crs: int, dict, or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str);
            proj (str or dict) or wkt (str). Only used if the data has no native CRS.
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
        name, catalog_name: str, optional
            Name of the dataset and catalog, optional for now.
        """
        if kwargs:
            warnings.warn(
                "Passing additional keyword arguments to be used by the "
                "GeoDatasetAdapter driver is deprecated and will be removed "
                "in a future version. Please use 'driver_kwargs' instead.",
                DeprecationWarning,
            )
            driver_kwargs.update(kwargs)
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
            name=name,
            catalog_name=catalog_name,
            provider=provider,
            version=version,
        )
        self.crs = crs

    def to_file(
        self,
        data_root,
        data_name,
        bbox=None,
        time_tuple=None,
        variables=None,
        driver=None,
        logger=logger,
        **kwargs,
    ):
        """Save a data slice to file.

        Parameters
        ----------
        data_root : str, Path
            Path to output folder
        data_name : str
            Name of output file without extension.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        driver : str, optional
            Driver to write file, e.g.: 'netcdf', 'zarr', by default None
        variables : list of str, optional
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.
        **kwargs
            Additional keyword arguments that are passed to the `to_zarr`
            function.

        Returns
        -------
        fn_out: str
            Absolute path to output file
        driver: str
            Name of driver to read data with, see
            :py:func:`~hydromt.data_catalog.DataCatalog.get_geodataset`
        """
        obj = self.get_data(
            bbox=bbox,
            time_tuple=time_tuple,
            variables=variables,
            logger=logger,
            single_var_as_array=variables is None,
        )

        read_kwargs = {}

        # much better for mem/storage/processing if dtypes are set correctly
        for name, coord in obj.coords.items():
            if coord.values.dtype != object:
                continue

            # not sure if coordinates values of different dtypes
            # are possible, but let's just hope users aren't
            # that mean for now.
            if isinstance(coord.values[0], str):
                obj[name] = obj[name].astype(str)

        if driver is None or driver == "netcdf":
            # always write netcdf
            driver = "netcdf"
            dvars = [obj.name] if isinstance(obj, xr.DataArray) else obj.vector.vars
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
            raise ValueError(f"GeoDataset: Driver {driver} unknown.")

        return fn_out, driver, read_kwargs

    def get_data(
        self,
        bbox=None,
        geom=None,
        buffer=0,
        predicate="intersects",
        variables=None,
        time_tuple=None,
        single_var_as_array=True,
        logger=logger,
    ):
        """Return a clipped, sliced and unified GeoDataset.

        For a detailed description see:
        :py:func:`~hydromt.data_catalog.DataCatalog.get_geodataset`
        """
        # load data
        fns = self._resolve_paths(variables, time_tuple)
        ds = self._read_data(fns, logger=logger)
        # rename variables and parse data and attrs
        ds = self._rename_vars(ds)
        ds = self._validate_spatial_coords(ds)
        ds = self._set_crs(ds, logger=logger)
        ds = self._set_nodata(ds)
        ds = self._shift_time(ds, logger=logger)
        # slice
        ds = GeoDatasetAdapter._slice_data(
            ds, variables, geom, bbox, buffer, predicate, time_tuple, logger=logger
        )
        # uniformize
        ds = self._apply_unit_conversion(ds, logger=logger)
        ds = self._set_metadata(ds)
        # return array if single var and single_var_as_array
        return self._single_var_as_array(ds, single_var_as_array, variables)

    def _resolve_paths(self, variables, time_tuple):
        # Extract storage_options from kwargs to instantiate fsspec object correctly
        so_kwargs = dict()
        if "storage_options" in self.driver_kwargs and self.driver == "zarr":
            so_kwargs = self.driver_kwargs["storage_options"]
            # For s3, anonymous connection still requires --no-sign-request profile to
            # read the data
            # setting environment variable works
            if "anon" in so_kwargs:
                os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
            else:
                os.environ["AWS_NO_SIGN_REQUEST"] = "NO"
        elif "storage_options" in self.driver_kwargs:
            raise NotImplementedError(
                "Remote (cloud) GeoDataset only supported with driver zarr."
            )

        # resolve paths
        fns = super()._resolve_paths(
            time_tuple=time_tuple, variables=variables, **so_kwargs
        )

        return fns

    def _read_data(self, fns, logger=logger):
        kwargs = self.driver_kwargs.copy()
        if len(fns) > 1 and self.driver in ["vector", "zarr"]:
            raise ValueError(
                f"GeoDataset: Reading multiple {self.driver} files is not supported."
            )
        logger.info(f"Reading {self.name} {self.driver} data from {self.path}")
        if self.driver in ["netcdf"]:
            ds = xr.open_mfdataset(fns, **kwargs)
        elif self.driver == "zarr":
            ds = xr.open_zarr(fns[0], **kwargs)
        elif self.driver == "vector":
            ds = io.open_geodataset(fn_locs=fns[0], crs=self.crs, **kwargs)
        else:
            raise ValueError(f"GeoDataset: Driver {self.driver} unknown")

        return ds

    def _rename_vars(self, ds):
        rm = {k: v for k, v in self.rename.items() if k in ds}
        ds = ds.rename(rm)
        return ds

    def _validate_spatial_coords(self, ds):
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
            raise ValueError(
                f"GeoDataset: No spatial geometry dimension found in data {self.path}"
            )
        return ds

    def _set_crs(self, ds, logger=logger):
        # set crs
        if ds.vector.crs is None and self.crs is not None:
            ds.vector.set_crs(self.crs)
        elif ds.vector.crs is None:
            raise ValueError(
                f"GeoDataset {self.name}: CRS not defined in data catalog or data."
            )
        elif self.crs is not None and ds.vector.crs != pyproj.CRS.from_user_input(
            self.crs
        ):
            logger.warning(
                f"GeoDataset {self.name}: CRS from data catalog does not match CRS of"
                " data. The original CRS will be used. Please check your data catalog."
            )
        return ds

    @staticmethod
    def _slice_data(
        ds,
        variables=None,
        geom=None,
        bbox=None,
        buffer=0,
        predicate="intersects",
        time_tuple=None,
        logger=logger,
    ):
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
            :py:func:`hydromt.gis_utils.filter_gdf` for details.
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
            variables = np.atleast_1d(variables).tolist()
            if len(variables) > 1 or len(ds.data_vars) > 1:
                mvars = [var not in ds.data_vars for var in variables]
                if any(mvars):
                    raise ValueError(f"GeoDataset: variables not found {mvars}")
                ds = ds[variables]
        if time_tuple is not None:
            ds = GeoDatasetAdapter._slice_temporal_dimension(
                ds, time_tuple, logger=logger
            )
        if geom is not None or bbox is not None:
            ds = GeoDatasetAdapter._slice_spatial_dimension(
                ds, geom, bbox, buffer, predicate, logger=logger
            )
        return ds

    @staticmethod
    def _slice_spatial_dimension(ds, geom, bbox, buffer, predicate, logger=logger):
        geom = gis_utils.parse_geom_bbox_buffer(geom, bbox, buffer)
        bbox_str = ", ".join([f"{c:.3f}" for c in geom.total_bounds])
        epsg = geom.crs.to_epsg()
        logger.debug(f"Clip {predicate} [{bbox_str}] (EPSG:{epsg})")
        ds = ds.vector.clip_geom(geom, predicate=predicate)
        if ds.vector.index.size == 0:
            raise IndexError("No data within spatial domain.")
        return ds

    def _shift_time(self, ds, logger=logger):
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
    def _slice_temporal_dimension(ds, time_tuple, logger=logger):
        if (
            "time" in ds.dims
            and ds["time"].size > 1
            and np.issubdtype(ds["time"].dtype, np.datetime64)
        ):
            logger.debug(f"Slicing time dim {time_tuple}")
            ds = ds.sel(time=slice(*time_tuple))
            if ds.time.size == 0:
                raise IndexError("GeoDataset: Time slice out of range.")
        return ds

    def _set_metadata(self, ds):
        if self.attrs:
            if isinstance(ds, xr.DataArray):
                ds.attrs.update(self.attrs[ds.name])
            else:
                for k in self.attrs:
                    ds[k].attrs.update(self.attrs[k])

        ds.attrs.update(self.meta)
        return ds

    def _set_nodata(self, ds):
        if self.nodata is not None:
            if not isinstance(self.nodata, dict):
                nodata = {k: self.nodata for k in ds.data_vars.keys()}
            else:
                nodata = self.nodata
            for k in ds.data_vars:
                mv = nodata.get(k, None)
                if mv is not None and ds[k].vector.nodata is None:
                    ds[k].vector.set_nodata(mv)
        return ds

    def _apply_unit_conversion(self, ds, logger=logger):
        unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
        unit_names = [k for k in unit_names if k in ds.data_vars]
        if len(unit_names) > 0:
            logger.debug(f"Convert units for {len(unit_names)} variables.")
        for name in list(set(unit_names)):  # unique
            m = self.unit_mult.get(name, 1)
            a = self.unit_add.get(name, 0)
            da = ds[name]
            attrs = da.attrs.copy()
            nodata_isnan = da.vector.nodata is None or np.isnan(da.vector.nodata)
            # nodata value is explicitly set to NaN in case no nodata value is provided
            nodata = np.nan if nodata_isnan else da.vector.nodata
            data_bool = ~np.isnan(da) if nodata_isnan else da != nodata
            ds[name] = xr.where(data_bool, da * m + a, nodata)
            ds[name].attrs.update(attrs)  # set original attributes
        return ds

    def detect_spatial_range(self, ds=None) -> geometry:
        """Detect spatial range."""
        if ds is None:
            ds = self.get_data()
        return box(*ds.vector.bounds)

    def detect_temporal_range(
        self, ds=None, time_dim_name="time"
    ) -> Tuple[datetime, datetime]:
        """Detect temporal range."""
        if ds is None:
            ds = self.get_data()
        try:
            time_range = ds[time_dim_name]
        except KeyError:
            time_range = ds.T.index
            if pd.api.types.is_numeric_dtype(time_range):
                # pd.to_datetime will simply parse ints etc.
                # which we don't want so we have to raise the error
                # ourselves
                raise KeyError("No time dimension found")
            else:
                time_range = pd.to_datetime(time_range)

        return (time_range.min().values, time_range.max().values)
