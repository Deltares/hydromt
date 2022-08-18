# -*- coding: utf-8 -*-
from os.path import join
from .. import gis_utils, io
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import warnings

from .data_adapter import DataAdapter, PREPROCESSORS
from ..raster import GEO_MAP_COORD

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "RasterDatasetAdapter",
]


class RasterDatasetAdapter(DataAdapter):
    _DEFAULT_DRIVER = "raster"
    _DRIVERS = {
        "nc": "netcdf",
        "zarr": "zarr",
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
        units={},
        meta={},
        placeholders={},
        **kwargs,
    ):
        """Initiates data adapter for geospatial raster data.

        This object contains all properties required to read supported raster files into
        a single unified RasterDataset, i.e. :py:class:`xarray.Dataset` with geospatial attributes.
        In addition it keeps meta data to be able to reproduce which data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source. If the dataset consists of multiple files, the path may
            contain {variable}, {year}, {month} placeholders as well as path search pattern
            using a '*' wildcard.
        driver: {'raster', 'netcdf', 'zarr', 'raster_tindex'}, optional
            Driver to read files with, for 'raster' :py:func:`~hydromt.io.open_mfraster`,
            for 'netcdf' :py:func:`xarray.open_mfdataset`, and for 'zarr' :py:func:`xarray.open_zarr`
            By default the driver is inferred from the file extension and falls back to
            'raster' if unknown.
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
            Metadata information of dataset, preferably containing the following keys:
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
        # TODO: see if the units argument can be solved with unit_mult/unit_add
        self.units = units

    def to_file(
        self,
        data_root,
        data_name,
        bbox=None,
        time_tuple=None,
        driver=None,
        variables=None,
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
            Driver to write file, e.g.: 'netcdf', 'zarr' or any gdal data type, by default None
        variables : list of str, optional
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.

        Returns
        -------
        fn_out: str
            Absolute path to output file
        driver: str
            Name of driver to read data with, see :py:func:`~hydromt.data_catalog.DataCatalog.get_rasterdataset`
        """

        try:
            obj = self.get_data(
                bbox=bbox, time_tuple=time_tuple, variables=variables, logger=logger
            )
        except IndexError as err:  # out of bounds
            logger.warning(str(err))
            return None, None

        if driver is None:
            driver = self.driver
            if driver in ["raster_tindex", "raster"]:
                # by default write 2D raster data to GeoTiff and 3D raster data to netcdf
                driver = "netcdf" if len(obj.dims) == 3 else "GTiff"
        # write using various writers
        if driver in ["netcdf"]:  # TODO complete list
            fn_out = join(data_root, f"{data_name}.nc")
            if "encoding" not in kwargs:
                dvars = [obj.name] if isinstance(obj, xr.DataArray) else obj.raster.vars
                kwargs.update(encoding={k: {"zlib": True} for k in dvars})
            obj.to_netcdf(fn_out, **kwargs)
        elif driver == "zarr":
            fn_out = join(data_root, f"{data_name}.zarr")
            obj.to_zarr(fn_out, **kwargs)
        elif driver not in gis_utils.GDAL_DRIVER_CODE_MAP.values():
            raise ValueError(f"RasterDataset: Driver {driver} unknown.")
        else:
            ext = gis_utils.GDAL_EXT_CODE_MAP.get(driver)
            if driver == "GTiff" and "compress" not in kwargs:
                kwargs.update(compress="lzw")  # default lzw compression
            if isinstance(obj, xr.DataArray):
                fn_out = join(data_root, f"{data_name}.{ext}")
                obj.raster.to_raster(fn_out, driver=driver, **kwargs)
            else:
                fn_out = join(data_root, data_name, "{variable}" + f".{ext}")
                obj.raster.to_mapstack(
                    join(data_root, data_name), driver=driver, **kwargs
                )
            driver = "raster"

        return fn_out, driver

    def get_data(
        self,
        bbox=None,
        geom=None,
        buffer=0,
        align=None,
        variables=None,
        time_tuple=None,
        single_var_as_array=True,
        logger=logger,
    ):
        """Returns a clipped, sliced and unified RasterDataset based on the properties
        of this RasterDatasetAdapter.

        For a detailed description see: :py:func:`~hydromt.data_catalog.DataCatalog.get_rasterdataset`
        """
        # If variable is string, convert to list
        if variables:
            variables = np.atleast_1d(variables).tolist()

        kwargs = self.kwargs.copy()
        fns = self.resolve_paths(time_tuple=time_tuple, variables=variables)

        # read using various readers
        if self.driver in ["netcdf"]:  # TODO complete list
            if "preprocess" in kwargs:
                preprocess = PREPROCESSORS.get(kwargs["preprocess"], None)
                kwargs.update(preprocess=preprocess)
            ds_out = xr.open_mfdataset(fns, decode_coords="all", **kwargs)
        elif self.driver == "zarr":
            if len(fns) > 1:
                raise ValueError(
                    "RasterDataset: Opening multiple zarr data files is not supported."
                )
            ds_out = xr.open_zarr(fns[0], **kwargs)
        elif self.driver == "raster_tindex":
            if np.issubdtype(type(self.nodata), np.number):
                kwargs.update(nodata=self.nodata)
            ds_out = io.open_raster_from_tindex(fns[0], bbox=bbox, geom=geom, **kwargs)
        elif self.driver == "raster":  # rasterio files
            if np.issubdtype(type(self.nodata), np.number):
                kwargs.update(nodata=self.nodata)
            ds_out = io.open_mfraster(fns, logger=logger, **kwargs)
        else:
            raise ValueError(f"RasterDataset: Driver {self.driver} unknown")
        if GEO_MAP_COORD in ds_out.data_vars:
            ds_out = ds_out.set_coords(GEO_MAP_COORD)

        # transpose dims to get y and x dim last
        x_dim = ds_out.raster.x_dim
        y_dim = ds_out.raster.y_dim
        ds_out = ds_out.transpose(..., y_dim, x_dim)

        # rename and select vars
        if variables and len(ds_out.raster.vars) == 1 and len(self.rename) == 0:
            rm = {ds_out.raster.vars[0]: variables[0]}
            if rm.keys() != rm.values():
                warnings.warn(
                    f"Automatic renaming of single var array will be deprecated, rename {rm} in the data catalog instead.",
                    DeprecationWarning,
                )
        else:
            rm = {k: v for k, v in self.rename.items() if k in ds_out}
        ds_out = ds_out.rename(rm)
        if variables is not None:
            if np.any([var not in ds_out.data_vars for var in variables]):
                raise ValueError(f"RasterDataset: Not all variables found: {variables}")
            ds_out = ds_out[variables]

        # clip tslice
        if (
            "time" in ds_out.dims
            and ds_out["time"].size > 1
            and np.issubdtype(ds_out["time"].dtype, np.datetime64)
        ):
            dt = self.unit_add.get("time", 0)
            if dt != 0:
                logger.debug(f"RasterDataset: Shifting time labels with {dt} sec.")
                ds_out["time"] = ds_out["time"] + pd.to_timedelta(dt, unit="s")
            if time_tuple is not None:
                logger.debug(f"RasterDataset: Slicing time dim {time_tuple}")
                ds_out = ds_out.sel({"time": slice(*time_tuple)})
            if ds_out.time.size == 0:
                raise IndexError(f"RasterDataset: Time slice out of range.")

        # set crs
        if ds_out.raster.crs is None and self.crs != None:
            ds_out.raster.set_crs(self.crs)
        elif ds_out.raster.crs is None:
            raise ValueError(
                "RasterDataset: The data has no CRS, set in RasterDatasetAdapter."
            )

        # clip
        epsg = ds_out.raster.crs.to_epsg()
        if geom is not None:
            bbox = geom.to_crs(4326).total_bounds
        if epsg != 4326 and bbox is not None and geom is None:
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
        elif epsg == 4326:
            w, e = np.asarray(ds_out.raster.bounds)[[0, 2]]
            if e > 180 or (bbox is not None and (bbox[0] < -180 or bbox[2] > 180)):
                x_dim = ds_out.raster.x_dim
                ds_out = gis_utils.meridian_offset(ds_out, x_dim, bbox).sortby(x_dim)
        if bbox is not None:
            err = f"RasterDataset: No data within spatial domain for {self.path}."
            try:
                bbox_str = ", ".join([f"{c:.3f}" for c in bbox])
                if geom is not None:
                    logger.debug(f"RasterDataset: Clip with geom - [{bbox_str}]")
                    ds_out = ds_out.raster.clip_geom(geom, buffer=buffer, align=align)
                elif bbox is not None:
                    logger.debug(f"RasterDataset: Clip with bbox - [{bbox_str}]")
                    ds_out = ds_out.raster.clip_bbox(bbox, buffer=buffer, align=align)
            except IndexError:
                raise IndexError(err)
            if ds_out.raster.xcoords.size == 0 or ds_out.raster.ycoords.size == 0:
                raise IndexError(err)

        # set nodata value
        if self.nodata is not None:
            if not isinstance(self.nodata, dict):
                nodata = {k: self.nodata for k in ds_out.data_vars.keys()}
            else:
                nodata = self.nodata
            for k in ds_out.data_vars:
                mv = nodata.get(k, None)
                if mv is not None and ds_out[k].raster.nodata is None:
                    ds_out[k].raster.set_nodata(mv)

        # unit conversion
        unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
        unit_names = [k for k in unit_names if k in ds_out.data_vars]
        if len(unit_names) > 0:
            logger.debug(
                f"RasterDataset: Convert units for {len(unit_names)} variables."
            )
        for name in list(set(unit_names)):  # unique
            m = self.unit_mult.get(name, 1)
            a = self.unit_add.get(name, 0)
            da = ds_out[name]
            attrs = da.attrs.copy()
            nodata_isnan = da.raster.nodata is None or np.isnan(da.raster.nodata)
            # nodata value is explicitly set to NaN in case no nodata value is provided
            nodata = np.nan if nodata_isnan else da.raster.nodata
            data_bool = ~np.isnan(da) if nodata_isnan else da != nodata
            ds_out[name] = xr.where(data_bool, da * m + a, nodata)
            ds_out[name].attrs.update(attrs)  # set original attributes
            ds_out[name].raster.set_nodata(nodata)  # reset nodata in case of change

        # unit attributes
        # TODO: can we solve this with unit conversion or otherwise generalize meta
        for k in self.units:
            ds_out[k].attrs.update(units=self.units[k])

        # return data array if single var
        if single_var_as_array and len(ds_out.raster.vars) == 1:
            ds_out = ds_out[ds_out.raster.vars[0]]

        # set meta data
        ds_out.attrs.update(self.meta)
        return ds_out
