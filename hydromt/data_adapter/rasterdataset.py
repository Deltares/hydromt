"""Implementation for the RasterDatasetAdapter."""
from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime
from os import PathLike
from os.path import join
from typing import Dict, NewType, Optional, Tuple, Union, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import xarray as xr
from rasterio.errors import RasterioIOError

from .. import gis_utils, io
from ..nodata import NoDataStrategy, _exec_strat
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
        driver: Optional[str] = None,
        filesystem: Optional[str] = None,
        crs: Optional[Union[int, str, dict]] = None,
        nodata: Optional[Union[dict, float, int]] = None,
        rename: Optional[dict] = None,
        unit_mult: Optional[dict] = None,
        unit_add: Optional[dict] = None,
        meta: Optional[dict] = None,
        attrs: Optional[dict] = None,
        extent: Optional[dict] = None,
        driver_kwargs: Optional[dict] = None,
        storage_options: Optional[dict] = None,
        zoom_levels: Optional[dict] = None,
        name: str = "",  # optional for now
        catalog_name: str = "",  # optional for now
        provider: Optional[str] = None,
        version: Optional[str] = None,
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
            search pattern using a ``*`` wildcard.
        driver: {'raster', 'netcdf', 'zarr', 'raster_tindex'}, optional
            Driver to read files with,
            for 'raster' :py:func:`~hydromt.io.open_mfraster`,
            for 'netcdf' :py:func:`xarray.open_mfdataset`,
            and for 'zarr' :py:func:`xarray.open_zarr`
            By default the driver is inferred from the file extension and falls back to
            'raster' if unknown.
        filesystem: str, optional
            Filesystem where the data is stored (local, cloud, http etc.).
            If None (default) the filesystem is inferred from the path.
            See :py:func:`fsspec.registry.known_implementations` for all options.
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
        extent: Extent(typed dict), Optional
            Dictionary describing the spatial and time range the dataset covers.
            should be of the form:
            -  "bbox": [xmin, ymin, xmax, ymax],
            -  "time_range": [start_datetime, end_datetime],
            data, and time_range should be inclusive on both sides.
        driver_kwargs, dict, optional
            Additional key-word arguments passed to the driver.
        storage_options: dict, optional
            Additional key-word arguments passed to the fsspec FileSystem object.
        zoomlevels: dict, optional
            Dictionary with zoom levels and associated resolution in the unit of the
            data CRS.
        name, catalog_name: str, optional
            Name of the dataset and catalog, optional.
        provider: str, optional
            A name to identifiy the specific provider of the dataset requested.
            if None is provided, the last added source will be used.
        version: str, optional
            A name to identifiy the specific version of the dataset requested.
            if None is provided, the last added source will be used.

        """
        driver_kwargs = driver_kwargs or {}
        extent = extent or {}
        if kwargs:
            warnings.warn(
                "Passing additional keyword arguments to be used by the "
                "RasterDatasetAdapter driver is deprecated and will be removed "
                "in a future version. Please use 'driver_kwargs' instead.",
                DeprecationWarning,
                stacklevel=2,
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
            storage_options=storage_options,
            name=name,
            catalog_name=catalog_name,
            provider=provider,
            version=version,
        )
        self.crs = crs
        # should be None or non-empty dict when initialized
        self.zoom_levels = zoom_levels
        self.extent = extent

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
        self.mark_as_used()  # mark used
        ds = self._read_data(fns, geom, bbox, cache_root, zoom_level, logger)
        # rename variables and parse data and attrs
        ds = self._rename_vars(ds)
        ds = self._validate_spatial_dims(ds)
        ds = self._set_crs(ds, logger)
        ds = self._set_nodata(ds)
        ds = self._shift_time(ds, logger)
        # slice data
        ds = RasterDatasetAdapter._slice_data(
            ds, variables, geom, bbox, buffer, align, time_tuple, logger=logger
        )
        # uniformize data
        ds = self._apply_unit_conversions(ds, logger)
        ds = self._set_metadata(ds)
        # return array if single var and single_var_as_array
        return self._single_var_as_array(ds, single_var_as_array, variables)

    def _resolve_paths(
        self,
        time_tuple: Optional[tuple] = None,
        variables: Optional[list] = None,
        zoom_level: Optional[int] = 0,
        geom: Optional[gpd.GeoSeries] = None,
        bbox: Optional[list] = None,
        logger=logger,
    ):
        if zoom_level is not None and "{zoom_level}" in self.path:
            zoom_level = self._parse_zoom_level(zoom_level, geom, bbox, logger=logger)

        # resolve path based on time, zoom level and/or variables
        fns = super()._resolve_paths(
            time_tuple=time_tuple,
            variables=variables,
            zoom_level=zoom_level,
        )
        return fns

    def _read_data(self, fns, geom, bbox, cache_root, zoom_level=None, logger=logger):
        kwargs = self.driver_kwargs.copy()

        # read using various readers
        logger.info(f"Reading {self.name} {self.driver} data from {self.path}")
        if self.driver == "netcdf":
            if "preprocess" in kwargs:
                preprocess = PREPROCESSORS.get(kwargs["preprocess"], None)
                kwargs.update(preprocess=preprocess)
            ds = xr.open_mfdataset(fns, decode_coords="all", **kwargs)
        elif self.driver == "zarr":
            preprocess = None
            if "preprocess" in kwargs:  # for zarr preprocess is done after reading
                preprocess = PREPROCESSORS.get(kwargs.pop("preprocess"), None)
            ds_lst = []
            for fn in fns:
                ds = xr.open_zarr(fn, **kwargs)
                if preprocess:
                    ds = preprocess(ds)  # type: ignore
                ds_lst.append(ds)
            ds = xr.merge(ds_lst)
        elif self.driver == "raster_tindex":
            if np.issubdtype(type(self.nodata), np.number):
                kwargs.update(nodata=self.nodata)
            ds = io.open_raster_from_tindex(fns[0], bbox=bbox, geom=geom, **kwargs)
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
            if zoom_level is not None and "{zoom_level}" not in self.path:
                zls_dict, crs = self._get_zoom_levels_and_crs(logger=logger)
                zoom_level = self._parse_zoom_level(
                    zoom_level, geom, bbox, zls_dict, crs, logger=logger
                )
                if isinstance(zoom_level, int):
                    kwargs.update(overview_level=zoom_level)
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

    def _set_crs(self, ds, logger=logger):
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
        handle_nodata=NoDataStrategy.RAISE,
        logger=logger,
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
        handle_nodata: NoDataStrategy, optional
            How to handle no data values, by default NoDataStrategy.RAISE

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
            ds = RasterDatasetAdapter._slice_temporal_dimension(
                ds,
                time_tuple,
                handle_nodata,
                logger=logger,
            )
        if geom is not None or bbox is not None:
            ds = RasterDatasetAdapter._slice_spatial_dimensions(
                ds,
                geom,
                bbox,
                buffer,
                align,
                handle_nodata,
                logger=logger,
            )
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
    def _slice_temporal_dimension(
        ds, time_tuple, handle_nodata=NoDataStrategy.RAISE, logger=logger
    ):
        if (
            "time" in ds.dims
            and ds["time"].size > 1
            and np.issubdtype(ds["time"].dtype, np.datetime64)
        ):
            if time_tuple is not None:
                logger.debug(f"Slicing time dim {time_tuple}")
                ds = ds.sel({"time": slice(*time_tuple)})
                if ds.time.size == 0:
                    _exec_strat("Time slice out of range.", handle_nodata, logger)
        return ds

    @staticmethod
    def _slice_spatial_dimensions(
        ds,
        geom,
        bbox,
        buffer,
        align,
        handle_nodata=NoDataStrategy.RAISE,
        logger=logger,
    ):
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
            logger.debug(f"Clip to [{bbox_str}] (epsg:{epsg}))")
            ds = ds.raster.clip_bbox(bbox, buffer=buffer, align=align)
            if np.any(np.array(ds.raster.shape) < 2):
                _exec_strat(
                    "RasterDataset: No data within spatial domain",
                    handle_nodata,
                    logger,
                )

        return ds

    def _apply_unit_conversions(self, ds, logger=logger):
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

    def _get_zoom_levels_and_crs(self, logger=logger):
        """Get zoom levels and crs from adapter or detect from tif file if missing."""
        if self.zoom_levels is not None and self.crs is not None:
            return self.zoom_levels, self.crs
        zoom_levels = {}
        try:
            with rasterio.open(self.path) as src:
                res = abs(src.res[0])
                crs = src.crs
                overviews = [src.overviews(i) for i in src.indexes]
                # check if identical
                if not all([o == overviews[0] for o in overviews]):
                    raise ValueError("Overviews are not identical across bands")
                # dict with overview level and corresponding resolution
                zls = [1] + overviews[0]
                zoom_levels = {i: res * zl for i, zl in enumerate(zls)}
        except RasterioIOError as e:
            logger.warning(f"IO error while detecting zoom levels: {e}")
        self.zoom_levels = zoom_levels
        self.crs = crs
        return zoom_levels, crs

    def _parse_zoom_level(
        self,
        zoom_level: Union[int, Tuple[Union[int, float], str]],
        geom: Optional[gpd.GeoSeries] = None,
        bbox: Optional[list] = None,
        zls_dict: Optional[Dict[int, float]] = None,
        dst_crs: pyproj.CRS = None,
        logger=logger,
    ) -> int:
        """Return overview level of data corresponding to zoom level.

        Parameters
        ----------
        zoom_level: int or tuple
            overview level or tuple with resolution and unit
        geom: gpd.GeoSeries, optional
            geometry to determine res if zoom_level or source in degree
        bbox: list, optional
            bbox to determine res if zoom_level or source in degree
        zls_dict: dict, optional
            dictionary with overview levels and corresponding resolution
        dst_crs: pyproj.CRS, optional
            destination crs to determine res if zoom_level tuple is provided
            with different unit than dst_crs
        """
        # check zoom level
        zls_dict = self.zoom_levels if zls_dict is None else zls_dict
        dst_crs = self.crs if dst_crs is None else dst_crs
        if zls_dict is None or len(zls_dict) == 0 or zoom_level is None:
            return None
        elif isinstance(zoom_level, int):
            if zoom_level not in zls_dict:
                raise ValueError(
                    f"Zoom level {zoom_level} not defined."
                    f"Select from {list(zls_dict.keys())}."
                )
            zl = zoom_level
            dst_res = zls_dict[zoom_level]
        elif (
            isinstance(zoom_level, tuple)
            and isinstance(zoom_level[0], (int, float))
            and isinstance(zoom_level[1], str)
            and len(zoom_level) == 2
            and dst_crs is not None
        ):
            src_res, src_res_unit = zoom_level
            # convert res if different unit than crs
            dst_crs = pyproj.CRS.from_user_input(dst_crs)
            dst_crs_unit = dst_crs.axis_info[0].unit_name
            dst_res = src_res
            if dst_crs_unit != src_res_unit:
                known_units = ["degree", "metre", "US survey foot", "meter", "foot"]
                if src_res_unit not in known_units:
                    raise TypeError(
                        f"zoom_level unit {src_res_unit} not understood;"
                        f" should be one of {known_units}"
                    )
                if dst_crs_unit not in known_units:
                    raise NotImplementedError(
                        f"no conversion available for {src_res_unit} to {dst_crs_unit}"
                    )
                conversions = {
                    "foot": 0.3048,
                    "metre": 1,  # official pyproj units
                    "US survey foot": 0.3048,  # official pyproj units
                }  # to meter
                if src_res_unit == "degree" or dst_crs_unit == "degree":
                    lat = 0
                    if bbox is not None:
                        lat = (bbox[1] + bbox[3]) / 2
                    elif geom is not None:
                        lat = geom.to_crs(4326).centroid.y.item()
                    conversions["degree"] = gis_utils.cellres(lat=lat)[1]
                fsrc = conversions.get(src_res_unit, 1)
                fdst = conversions.get(dst_crs_unit, 1)
                dst_res = src_res * fsrc / fdst
            # find nearest zoom level
            eps = 1e-5  # allow for rounding errors
            zls = list(zls_dict.keys())
            smaller = [x < (dst_res + eps) for x in zls_dict.values()]
            zl = zls[-1] if all(smaller) else zls[max(smaller.index(False) - 1, 0)]
        elif dst_crs is None:
            raise ValueError("No CRS defined, hence no zoom level can be determined.")
        else:
            raise TypeError(f"zoom_level not understood: {type(zoom_level)}")
        logger.debug(f"Parsed zoom_level {zl} ({dst_res:.2f})")
        return zl

    def get_bbox(self, detect=True) -> Tuple[Tuple[float, float, float, float], int]:
        """Return the bounding box and espg code of the dataset.

        if the bounding box is not set and detect is True,
        :py:meth:`hydromt.RasterdatasetAdapter.detect_bbox` will be used to detect it.

        Parameters
        ----------
        detect: bool, Optional
            whether to detect the bounding box if it is not set. If False, and it's not
            set None will be returned.

        Returns
        -------
        bbox: Tuple[np.float64,np.float64,np.float64,np.float64]
            the bounding box coordinates of the data. coordinates are returned as
            [xmin,ymin,xmax,ymax]
        crs: int
            The ESPG code of the CRS of the coordinates returned in bbox
        """
        bbox = self.extent.get("bbox", None)
        crs = cast(int, self.crs)
        if bbox is None and detect:
            bbox, crs = self.detect_bbox()

        return bbox, crs

    def get_time_range(self, detect=True):
        """Detect the time range of the dataset.

        if the time range is not set and detect is True,
        :py:meth:`hydromt.RasterdatasetAdapter.detect_time_range` will be used
        to detect it.


        Parameters
        ----------
        detect: bool, Optional
            whether to detect the time range if it is not set. If False, and it's not
            set None will be returned.

        Returns
        -------
        range: Tuple[np.datetime64, np.datetime64]
            A tuple containing the start and end of the time dimension. Range is
            inclusive on both sides.
        """
        time_range = self.extent.get("time_range", None)
        if time_range is None and detect:
            time_range = self.detect_time_range()

        return time_range

    def detect_bbox(
        self,
        ds=None,
    ) -> Tuple[Tuple[float, float, float, float], int]:
        """Detect the bounding box and crs of the dataset.

        If no dataset is provided, it will be fetched according to the settings in the
        adapter. also see :py:meth:`hydromt.RasterdatasetAdapter.get_data`. the
        coordinates are in the CRS of the dataset itself, which is also returned
        alongside the coordinates.


        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the bounding box of.
            If none is provided, :py:meth:`hydromt.RasterdatasetAdapter.get_data`
            will be used to fetch the it before detecting.

        Returns
        -------
        bbox: Tuple[np.float64,np.float64,np.float64,np.float64]
            the bounding box coordinates of the data. coordinates are returned as
            [xmin,ymin,xmax,ymax]
        crs: int
            The ESPG code of the CRS of the coordinates returned in bbox
        """
        if ds is None:
            ds = self.get_data()
        crs = ds.raster.crs.to_epsg()
        bounds = ds.raster.bounds

        return bounds, crs

    def detect_time_range(self, ds=None) -> Tuple[datetime, datetime]:
        """Detect the temporal range of the dataset.

        If no dataset is provided, it will be fetched accodring to the settings in the
        addapter. also see :py:meth:`hydromt.RasterdatasetAdapter.get_data`.

        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the time range of. It must have a time dimentsion set.
            If none is provided, :py:meth:`hydromt.RasterdatasetAdapter.get_data`
            will be used to fetch the it before detecting.

        Returns
        -------
        range: Tuple[np.datetime64, np.datetime64]
            A tuple containing the start and end of the time dimension. Range is
            inclusive on both sides.
        """
        if ds is None:
            ds = self.get_data()
        return (
            ds[ds.raster.time_dim].min().values,
            ds[ds.raster.time_dim].max().values,
        )
