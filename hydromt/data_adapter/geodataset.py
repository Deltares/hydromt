# -*- coding: utf-8 -*-
from os.path import join
from .. import gis_utils, io
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

from .data_adapter import DataAdapter
from ..raster import GEO_MAP_COORD

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "GeoDatasetAdapter",
]


class GeoDatasetAdapter(DataAdapter):
    _DEFAULT_DRIVER = "vector"
    _DRIVERS = {
        "nc": "netcdf",
    }

    def __init__(
        self,
        path,
        driver=None,
        crs=None,
        nodata=None,
        rename={},
        unit_mult={},
        unit_add={},
        meta={},
        placeholders={},
        **kwargs,
    ):
        """Initiates data adapter for geospatial timeseries data.

        This object contains all properties required to read supported files into
        a single unified GeoDataset, i.e. :py:class:`xarray.Dataset` with geospatial point
        geometries. In addition it keeps meta data to be able to reproduce which data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source. If the dataset consists of multiple files, the path may
            contain {variable}, {year}, {month} placeholders as well as path search pattern
            using a '*' wildcard.
        driver: {'vector', 'netcdf', 'zarr'}, optional
            Driver to read files with, for 'vector' :py:func:`~hydromt.io.open_geodataset`,
            for 'netcdf' :py:func:`xarray.open_mfdataset`.
            By default the driver is inferred from the file extension and falls back to
            'vector' if unknown.
        crs: int, dict, or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str); proj (str or dict)
            or wkt (str). Only used if the data has no native CRS.
        nodata: float, int, optional
            Missing value number. Only used if the data has no native missing value.
            Nodata values can be differentiated between variables using a dictionary.
        rename: dict, optional
            Mapping of native data source variable to output source variable name as
            required by hydroMT.
        unit_mult, unit_add: dict, optional
            Scaling multiplication and addition to change to map from the native data unit
            to the output data unit as required by hydroMT.
        meta: dict, optional
            Metadata information of dataset, prefably containing the following keys:
            {'source_version', 'source_url', 'source_license', 'paper_ref', 'paper_doi', 'category'}
        placeholders: dict, optional
            Placeholders to expand yaml entry to multiple entries (name and path) based on placeholder values
        **kwargs
            Additional key-word arguments passed to the driver.
        """
        super().__init__(
            path=path,
            driver=driver,
            crs=crs,
            nodata=nodata,
            rename=rename,
            unit_mult=unit_mult,
            unit_add=unit_add,
            meta=meta,
            placeholders=placeholders,
            **kwargs,
        )

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

        Returns
        -------
        fn_out: str
            Absolute path to output file
        driver: str
            Name of driver to read data with, see :py:func:`~hydromt.data_catalog.DataCatalog.get_geodataset`
        """
        obj = self.get_data(
            bbox=bbox, time_tuple=time_tuple, variables=variables, logger=logger
        )
        if obj.vector.index.size == 0 or ("time" in obj.coords and obj.time.size == 0):
            return None, None

        if driver is None or driver == "netcdf":
            # always write netcdf
            driver = "netcdf"
            fn_out = join(data_root, f"{data_name}.nc")
            dvars = [obj.name] if isinstance(obj, xr.DataArray) else obj.vector.vars
            encoding = {k: {"zlib": True} for k in dvars}
            obj.to_netcdf(fn_out, encoding=encoding)
        elif driver == "zarr":
            fn_out = join(data_root, f"{data_name}.zarr")
            obj.to_zarr(fn_out, **kwargs)
        else:
            raise ValueError(f"GeoDataset: Driver {driver} unknown.")

        return fn_out, driver

    def get_data(
        self,
        bbox=None,
        geom=None,
        buffer=0,
        variables=None,
        time_tuple=None,
        single_var_as_array=True,
        logger=logger,
    ):
        """Returns a clipped, sliced and unified GeoDataset based on the properties
        of this GeoDatasetAdapter.

        For a detailed description see: :py:func:`~hydromt.data_catalog.DataCatalog.get_geodataset`
        """
        # If variable is string, convert to list
        if variables:
            variables = np.atleast_1d(variables).tolist()

        kwargs = self.kwargs.copy()
        fns = self.resolve_paths(time_tuple=time_tuple, variables=variables)

        # parse geom, bbox and buffer arguments
        clip_str = ""
        if geom is None and bbox is not None:
            # convert bbox to geom with crs EPGS:4326 to apply buffer later
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
            clip_str = " and clip to bbox (epsg:4326)"
        elif geom is not None:
            clip_str = f" and clip to geom (epsg:{geom.crs.to_epsg():d})"
        if geom is not None:
            # make sure geom is projected > buffer in meters!
            if buffer > 0 and geom.crs.is_geographic:
                geom = geom.to_crs(3857)
            geom = geom.buffer(buffer)
            bbox_str = ", ".join([f"{c:.3f}" for c in geom.total_bounds])
            clip_str = f"{clip_str} [{bbox_str}]"
        if kwargs.pop("within", False):  # for backward compatibility
            kwargs.update(predicate="contains")

        # read and clip
        logger.info(f"GeoDataset: Read {self.driver} data{clip_str}.")
        if self.driver in ["netcdf"]:
            ds_out = xr.open_mfdataset(fns, **kwargs)
        elif self.driver == "zarr":
            if len(fns) > 1:
                raise ValueError(
                    "GeoDataset: Opening multiple zarr data files is not supported."
                )
            ds_out = xr.open_zarr(fns[0], **kwargs)
        elif self.driver == "vector":
            # read geodataset from point + time series file
            ds_out = io.open_geodataset(
                fn_locs=fns[0], geom=geom, crs=self.crs, **kwargs
            )
            geom = None  # already clipped
        else:
            raise ValueError(f"GeoDataset: Driver {self.driver} unknown")
        if GEO_MAP_COORD in ds_out.data_vars:
            ds_out = ds_out.set_coords(GEO_MAP_COORD)

        # rename and select vars
        if variables and len(ds_out.vector.vars) == 1 and len(self.rename) == 0:
            rm = {ds_out.vector.vars[0]: variables[0]}
        else:
            rm = {k: v for k, v in self.rename.items() if k in ds_out}
        ds_out = ds_out.rename(rm)
        # check spatial dims and make sure all are set as coordinates
        try:
            ds_out.vector.set_spatial_dims()
            idim = ds_out.vector.index_dim
            if idim not in ds_out:  # set coordinates for index dimension if missing
                ds_out[idim] = xr.IndexVariable(idim, np.arange(ds_out.dims[idim]))
            coords = [ds_out.vector.x_dim, ds_out.vector.y_dim, idim]
            ds_out = ds_out.set_coords(coords)
        except ValueError:
            raise ValueError(f"GeoDataset: No spatial coords found in data {self.path}")
        if variables is not None:
            if np.any([var not in ds_out.data_vars for var in variables]):
                raise ValueError(f"GeoDataset: Not all variables found: {variables}")
            ds_out = ds_out[variables]

        # set crs
        if ds_out.vector.crs is None and self.crs != None:
            ds_out.vector.set_crs(self.crs)
        if ds_out.vector.crs is None:
            raise ValueError(
                "GeoDataset: The data has no CRS, set in GeoDatasetAdapter."
            )

        # clip
        if geom is not None:
            bbox = geom.to_crs(4326).total_bounds
        if ds_out.vector.crs.to_epsg() == 4326:
            w, e = (
                ds_out.vector.xcoords.values.min(),
                ds_out.vector.xcoords.values.max(),
            )
            if e > 180 or (bbox is not None and (bbox[0] < -180 or bbox[2] > 180)):
                ds_out = gis_utils.meridian_offset(ds_out, ds_out.vector.x_dim, bbox)
        if geom is not None:
            predicate = kwargs.pop("predicate", "intersects")
            ds_out = ds_out.vector.clip_geom(geom, predicate=predicate)
        if ds_out.vector.index.size == 0:
            logger.warning(
                f"GeoDataset: No data within spatial domain for {self.path}."
            )

        # clip tslice
        if (
            "time" in ds_out.dims
            and ds_out["time"].size > 1
            and np.issubdtype(ds_out["time"].dtype, np.datetime64)
        ):
            dt = self.unit_add.get("time", 0)
            if dt != 0:
                logger.debug(f"GeoDataset: Shifting time labels with {dt} sec.")
                ds_out["time"] = ds_out["time"] + pd.to_timedelta(dt, unit="s")
            if time_tuple is not None:
                logger.debug(f"GeoDataset: Slicing time dim {time_tuple}")
                ds_out = ds_out.sel(time=slice(*time_tuple))
            if ds_out.time.size == 0:
                logger.warning(f"GeoDataset: Time slice out of range.")
                drop_vars = [v for v in ds_out.data_vars if "time" in ds_out[v].dims]
                ds_out = ds_out.drop(drop_vars)

        # set nodata value
        if self.nodata is not None:
            if not isinstance(self.nodata, dict):
                nodata = {k: self.nodata for k in ds_out.data_vars.keys()}
            else:
                nodata = self.nodata
            for k in ds_out.data_vars:
                mv = nodata.get(k, None)
                if mv is not None and ds_out[k].vector.nodata is None:
                    ds_out[k].vector.set_nodata(mv)

        # unit conversion
        unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
        unit_names = [k for k in unit_names if k in ds_out.data_vars]
        if len(unit_names) > 0:
            logger.debug(f"GeoDataset: Convert units for {len(unit_names)} variables.")
        for name in list(set(unit_names)):  # unique
            m = self.unit_mult.get(name, 1)
            a = self.unit_add.get(name, 0)
            da = ds_out[name]
            attrs = da.attrs.copy()
            nodata_isnan = da.vector.nodata is None or np.isnan(da.vector.nodata)
            # nodata value is explicitly set to NaN in case no nodata value is provided
            nodata = np.nan if nodata_isnan else da.vector.nodata
            data_bool = ~np.isnan(da) if nodata_isnan else da != nodata
            ds_out[name] = xr.where(data_bool, da * m + a, nodata)
            ds_out[name].attrs.update(attrs)  # set original attributes

        # return data array if single var
        if single_var_as_array and len(ds_out.vector.vars) == 1:
            ds_out = ds_out[ds_out.vector.vars[0]]

        # set meta data
        ds_out.attrs.update(self.meta)

        return ds_out
