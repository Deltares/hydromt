"""Implementation for the RasterDatasetAdapter."""
from __future__ import annotations

import logging
import os
import warnings
from os import PathLike
from os.path import join
from typing import NewType, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import xarray as xr

from .. import gis_utils, io
from ..raster import GEO_MAP_COORD
from .caching import cache_vrt_tiles
from .data_adapter import PREPROCESSORS, DataAdapter

logger = logging.getLogger(__name__)

__all__ = ["RasterDatasetAdapter", "RasterDatasetSource"]

RasterDatasetSource = NewType("RasterDatasetSource", Union[str, PathLike])


class RasterDatasetAdapter(DataAdapter):

    """Implementation for the RasterDatasetAdapter."""

    _DEFAULT_DRIVER = "raster"
    _DRIVERS = {
        "nc": "netcdf",
        "zarr": "zarr",
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
        zoom_levels: dict = {},
        name: str = "",  # optional for now
        catalog_name: str = "",  # optional for now
        provider=None,
        version=None,
        **kwargs,
    ):
        """Initiate data adapter for geospatial raster data.

        This object contains all properties required to read supported raster files into
        a single unified RasterDataset, i.e. :py:class:`xarray.Dataset` with geospatial
        attributes. In addition it keeps meta data to be able to reproduce
        which data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source. If the dataset consists of multiple files, the path may
            contain {variable}, {year}, {month} placeholders as well as path
            search pattern using a '*' wildcard.
        driver: {'raster', 'netcdf', 'zarr', 'raster_tindex'}, optional
            Driver to read files with,
            for 'raster' :py:func:`~hydromt.io.open_mfraster`,
            for 'netcdf' :py:func:`xarray.open_mfdataset`,
            and for 'zarr' :py:func:`xarray.open_zarr`
            By default the driver is inferred from the file extension and falls back to
            'raster' if unknown.
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
            {'source_version', 'source_url', 'source_license',
            'paper_ref', 'paper_doi', 'category'}
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
        zoomlevels: dict, optional
            Dictionary with zoom levels and associated resolution in the unit of the
            data CRS.

        """
        if kwargs:
            warnings.warn(
                "Passing additional keyword arguments to be used by the "
                "RasterDatasetAdapter driver is deprecated and will be removed "
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
        self.zoom_levels = zoom_levels

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
            Driver to write file, e.g.: 'netcdf', 'zarr' or any gdal data type,
            by default None
        variables : list of str, optional
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.
        logger : logger object, optional
            The logger object used for logging messages. If not provided, the default
            logger will be used.
        **kwargs
            Additional keyword arguments that are passed to the `to_netcdf`
            function.

        Returns
        -------
        fn_out: str
            Absolute path to output file
        driver: str
            Name of driver to read data with, see
            :py:func:`~hydromt.data_catalog.DataCatalog.get_rasterdataset`
        kwargs: dict
            the additional kwyeord arguments that were passed to `to_netcdf`
        """
        obj = self.get_data(
            bbox=bbox,
            time_tuple=time_tuple,
            variables=variables,
            logger=logger,
            single_var_as_array=variables is None,
        )

        read_kwargs = {}
        if driver is None:
            # by default write 2D raster data to GeoTiff and 3D raster data to netcdf
            driver = "netcdf" if len(obj.dims) == 3 else "GTiff"
        # write using various writers
        if driver == "netcdf":
            dvars = [obj.name] if isinstance(obj, xr.DataArray) else obj.raster.vars
            if variables is None:
                encoding = {k: {"zlib": True} for k in dvars}
                fn_out = join(data_root, f"{data_name}.nc")
                obj.to_netcdf(fn_out, encoding=encoding, **kwargs)
            else:  # save per variable
                if not os.path.isdir(join(data_root, data_name)):
                    os.makedirs(join(data_root, data_name))
                for var in dvars:
                    fn_out = join(data_root, data_name, f"{var}.nc")
                    obj[var].to_netcdf(fn_out, encoding={var: {"zlib": True}}, **kwargs)
                fn_out = join(data_root, data_name, "{variable}.nc")
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

        return fn_out, driver, read_kwargs

    def get_data(
        self,
        bbox=None,
        geom=None,
        buffer=0,
        zoom_level=None,
        align=None,
        variables=None,
        time_tuple=None,
        single_var_as_array=True,
        cache_root=None,
        logger=logger,
    ):
        """Return a clipped, sliced and unified RasterDataset.

        For a detailed description see:
        :py:func:`~hydromt.data_catalog.DataCatalog.get_rasterdataset`
        """
        # load data
        fns = self._resolve_paths(time_tuple, variables, zoom_level, geom, bbox, logger)
        ds = self._read_data(fns, geom, bbox, logger, cache_root)
        # rename variables and parse data and attrs
        ds = self._rename_vars(ds)
        ds = self._validate_spatial_dims(ds)
        ds = self._set_crs(ds)
        ds = self._set_nodata(ds)
        ds = self._shift_time(ds)
        # slice data
        ds = self._slice_data(ds, variables, geom, bbox, buffer, align, time_tuple)
        # uniformize data
        ds = self._apply_unit_conversions(ds, logger)
        ds = self._set_metadata(ds)
        # return array if single var and single_var_as_array
        return self._single_var_as_array(ds, single_var_as_array, variables)

    def _resolve_paths(
        self,
        time_tuple: tuple = None,
        variables: list = None,
        zoom_level: int = 0,
        geom: gpd.GeoSeries = None,
        bbox: list = None,
        logger=logger,
    ):
        # Extract storage_options from kwargs to instantiate fsspec object correctly
        so_kwargs = dict()
        if "storage_options" in self.driver_kwargs:
            so_kwargs = self.driver_kwargs["storage_options"]
            # For s3, anonymous connection still requires --no-sign-request profile to
            # read the data setting environment variable works
            if "anon" in so_kwargs:
                os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
            else:
                os.environ["AWS_NO_SIGN_REQUEST"] = "NO"

        # parse zoom level (raster only)
        if len(self.zoom_levels) > 0:
            zoom_level = self._parse_zoom_level(zoom_level, geom, bbox, logger=logger)

        # resolve path based on time, zoom level and/or variables
        fns = super()._resolve_paths(
            time_tuple=time_tuple,
            variables=variables,
            zoom_level=zoom_level,
            logger=logger,
            **so_kwargs,
        )

        return fns

    def _read_data(self, fns, geom, bbox, logger, cache_root):
        kwargs = self.driver_kwargs.copy()

        # zarr can use storage options directly, the rest should be converted to
        # file-like objects
        if "storage_options" in kwargs and self.driver == "raster":
            storage_options = kwargs.pop("storage_options")
            fs = self.get_filesystem(**storage_options)
            fns = [fs.open(f) for f in fns]

        # read using various readers
        if self.driver == "netcdf":
            if self.filesystem == "local":
                if "preprocess" in kwargs:
                    preprocess = PREPROCESSORS.get(kwargs["preprocess"], None)
                    kwargs.update(preprocess=preprocess)
                ds = xr.open_mfdataset(fns, decode_coords="all", **kwargs)
            else:
                raise NotImplementedError(
                    "Remote (cloud) RasterDataset not supported with driver netcdf."
                )
        elif self.driver == "zarr":
            if "preprocess" in kwargs:  # for zarr preprocess is done after reading
                preprocess = PREPROCESSORS.get(kwargs.pop("preprocess"), None)
                do_preprocess = True
            else:
                do_preprocess = False
            ds_lst = []
            for fn in fns:
                ds = xr.open_zarr(fn, **kwargs)
                if do_preprocess:
                    ds = preprocess(ds)
                ds_lst.append(ds)
            ds = xr.merge(ds_lst)
        elif self.driver == "raster_tindex":
            if self.filesystem == "local":
                if np.issubdtype(type(self.nodata), np.number):
                    kwargs.update(nodata=self.nodata)
                ds = io.open_raster_from_tindex(fns[0], bbox=bbox, geom=geom, **kwargs)
            else:
                raise NotImplementedError(
                    "Remote (cloud) RasterDataset not supported "
                    "with driver raster_tindex."
                )
        elif self.driver == "raster":  # rasterio files
            if cache_root is not None and all([str(fn).endswith(".vrt") for fn in fns]):
                cache_dir = join(cache_root, self.catalog_name, self.name)
                fns_cached = []
                for fn in fns:
                    fn1 = cache_vrt_tiles(
                        fn, geom=geom, cache_dir=cache_dir, logger=logger
                    )
                    fns_cached.append(fn1)
                fns = fns_cached
            if np.issubdtype(type(self.nodata), np.number):
                kwargs.update(nodata=self.nodata)
            ds = io.open_mfraster(fns, logger=logger, **kwargs)
        else:
            raise ValueError(f"RasterDataset: Driver {self.driver} unknown")

        return ds

    def _rename_vars(self, ds):
        rm = {k: v for k, v in self.rename.items() if k in ds}
        ds = ds.rename(rm)
        return ds

    def _validate_spatial_dims(self, ds):
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

    def _set_crs(self, ds):
        # set crs
        if ds.raster.crs is None and self.crs is not None:
            ds.raster.set_crs(self.crs)
        elif ds.raster.crs is None:
            raise ValueError(
                f"RasterDataset {self.name}: CRS not defined in data catalog or data."
            )
        elif self.crs is not None and ds.raster.crs != pyproj.CRS.from_user_input(
            self.crs
        ):
            logger.warning(
                f"RasterDataset {self.name}: CRS from data catalog does not match CRS "
                " of data. The original CRS will be used. Please check your catalog."
            )
        return ds

    @staticmethod
    def _slice_data(
        ds,
        variables=None,
        geom=None,
        bbox=None,
        buffer=0,
        align=None,
        time_tuple=None,
    ):
        """Return a RasterDataset sliced in both spatial and temporal dimensions.

        Arguments
        ---------
        ds : xarray.Dataset or xarray.DataArray
            The RasterDataset to slice.
        variables : list of str, optional
            Names of variables to return. By default all dataset variables
        geom : geopandas.GeoDataFrame/Series, optional
            A geometry defining the area of interest.
        bbox : array-like of floats, optional
            (xmin, ymin, xmax, ymax) bounding box of area of interest
            (in WGS84 coordinates).
        buffer : int, optional
            Buffer around the `bbox` or `geom` area of interest in pixels. By default 0.
        align : float, optional
            Resolution to align the bounding box, by default None
        time_tuple : Tuple of datetime, optional
            A tuple consisting of the lower and upper bounds of time that the
            result should contain

        Returns
        -------
        ds : xarray.Dataset
            The sliced RasterDataset.
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
                    raise ValueError(f"RasterDataset: variables not found {mvars}")
                ds = ds[variables]
        if time_tuple is not None:
            ds = RasterDatasetAdapter._slice_temporal_dimension(ds, time_tuple)
        if geom is not None or bbox is not None:
            ds = RasterDatasetAdapter._slice_spatial_dimensions(
                ds, geom, bbox, buffer, align
            )
        return ds

    def _shift_time(self, ds):
        dt = self.unit_add.get("time", 0)
        if (
            dt != 0
            and "time" in ds.dims
            and ds["time"].size > 1
            and np.issubdtype(ds["time"].dtype, np.datetime64)
        ):
            logger.debug(f"GeoDataset: Shifting time labels with {dt} sec.")
            ds["time"] = ds["time"] + pd.to_timedelta(dt, unit="s")
        elif dt != 0:
            logger.warning(
                "GeoDataset: Time shift not applied, time dimension not found."
            )
        return ds

    @staticmethod
    def _slice_temporal_dimension(ds, time_tuple):
        if (
            "time" in ds.dims
            and ds["time"].size > 1
            and np.issubdtype(ds["time"].dtype, np.datetime64)
        ):
            if time_tuple is not None:
                logger.debug(f"RasterDataset: Slicing time dim {time_tuple}")
                ds = ds.sel({"time": slice(*time_tuple)})
                if ds.time.size == 0:
                    raise IndexError("RasterDataset: Time slice out of range.")
        return ds

    @staticmethod
    def _slice_spatial_dimensions(ds, geom, bbox, buffer, align):
        # make sure bbox is in data crs
        crs = ds.raster.crs
        epsg = crs.to_epsg()  # this could return None
        if geom is not None:
            bbox = geom.to_crs(crs).total_bounds
        elif epsg != 4326 and bbox is not None:
            crs4326 = pyproj.CRS.from_epsg(4326)
            bbox = rasterio.warp.transform_bounds(crs4326, crs, *bbox)
        # work with 4326 data that is defined at 0-360 degrees longtitude
        if epsg == 4326:
            e = ds.raster.bounds[2]
            if e > 180 or (bbox is not None and (bbox[0] < -180 or bbox[2] > 180)):
                x_dim = ds.raster.x_dim
                ds = gis_utils.meridian_offset(ds, x_dim, bbox).sortby(x_dim)

        # clip with bbox
        if bbox is not None:
            bbox_str = ", ".join([f"{c:.3f}" for c in bbox])
            logger.debug(f"RasterDataset: Clip to - [{bbox_str}] (epsg:{epsg}))")
            ds = ds.raster.clip_bbox(bbox, buffer=buffer, align=align)
            if np.any(np.array(ds.raster.shape) < 2):
                raise IndexError("RasterDataset: No data within spatial domain.")

        return ds

    def _apply_unit_conversions(self, ds, logger):
        unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
        unit_names = [k for k in unit_names if k in ds.data_vars]
        if len(unit_names) > 0:
            logger.debug(
                f"RasterDataset: Convert units for {len(unit_names)} variables."
            )
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

    def _set_nodata(self, ds):
        # set nodata value
        if self.nodata is not None:
            if not isinstance(self.nodata, dict):
                nodata = {k: self.nodata for k in ds.data_vars.keys()}
            else:
                nodata = self.nodata
            for k in ds.data_vars:
                mv = nodata.get(k, None)
                if mv is not None and ds[k].raster.nodata is None:
                    ds[k].raster.set_nodata(mv)
        return ds

    def _set_metadata(self, ds):
        # unit attributes
        for k in self.attrs:
            ds[k].attrs.update(self.attrs[k])
        # set meta data
        ds.attrs.update(self.meta)
        return ds

    def _parse_zoom_level(
        self,
        zoom_level: int | tuple = None,
        geom: gpd.GeoSeries = None,
        bbox: list = None,
        logger=logger,
    ) -> int:
        """Return nearest smaller zoom level.

        Based on zoom resolutions defined in data catalog.
        """
        # common pyproj crs axis units
        known_units = ["degree", "metre", "US survey foot"]
        if self.zoom_levels is None or len(self.zoom_levels) == 0:
            logger.warning("No zoom levels available, default to zero")
            return 0
        zls = list(self.zoom_levels.keys())
        if zoom_level is None:  # return first zoomlevel (assume these are ordered)
            return next(iter(zls))
        # parse zoom_level argument
        if (
            isinstance(zoom_level, tuple)
            and isinstance(zoom_level[0], (int, float))
            and isinstance(zoom_level[1], str)
            and len(zoom_level) == 2
        ):
            res, unit = zoom_level
            # covert 'meter' and foot to official pyproj units
            unit = {"meter": "metre", "foot": "US survey foot"}.get(unit, unit)
            if unit not in known_units:
                raise TypeError(
                    f"zoom_level unit {unit} not understood;"
                    f" should be one of {known_units}"
                )
        elif not isinstance(zoom_level, int):
            raise TypeError(
                f"zoom_level argument not understood: {zoom_level}; should be a float"
            )
        else:
            return zoom_level
        if self.crs:
            # convert res if different unit than crs
            crs = pyproj.CRS.from_user_input(self.crs)
            crs_unit = crs.axis_info[0].unit_name
            if crs_unit != unit and crs_unit not in known_units:
                raise NotImplementedError(
                    f"no conversion available for {unit} to {crs_unit}"
                )
            if unit != crs_unit:
                lat = 0
                if bbox is not None:
                    lat = (bbox[1] + bbox[3]) / 2
                elif geom is not None:
                    lat = geom.to_crs(4326).centroid.y.item()
                conversions = {
                    "degree": np.hypot(*gis_utils.cellres(lat=lat)),
                    "US survey foot": 0.3048,
                }
                res = res * conversions.get(unit, 1) / conversions.get(crs_unit, 1)
        # find nearest smaller zoomlevel
        eps = 1e-5  # allow for rounding errors
        smaller = [x < (res + eps) for x in self.zoom_levels.values()]
        zl = zls[-1] if all(smaller) else zls[max(smaller.index(False) - 1, 0)]
        logger.info(f"Getting data for zoom_level {zl} based on res {zoom_level}")
        return zl
