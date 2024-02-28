"""Implementation for the RasterDatasetAdapter."""
from __future__ import annotations

import os
from datetime import datetime
from logging import Logger, getLogger
from os.path import basename, join, splitext
from typing import Dict, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import pyproj
import rasterio
import xarray as xr
from pydantic import PrivateAttr
from pyproj.exceptions import CRSError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from pystac import MediaType
from rasterio.errors import RasterioIOError

from hydromt._typing import (
    Bbox,
    Data,
    ErrorHandleMethod,
    Geom,
    GeomBuffer,
    NoDataException,
    NoDataStrategy,
    RasterDatasetSource,
    StrPath,
    TimeRange,
    TotalBounds,
    Variables,
    _exec_nodata_strat,
)
from hydromt.data_adapter.data_adapter_base import DataAdapterBase
from hydromt.data_adapter.utils import has_no_data
from hydromt.data_sources import RasterDataSource
from hydromt.gis import utils
from hydromt.gis.raster import GEO_MAP_COORD

logger = getLogger(__name__)

__all__ = ["RasterDatasetAdapter", "RasterDatasetSource"]


class RasterDatasetAdapter(DataAdapterBase):
    """Implementation for the RasterDatasetAdapter."""

    source: RasterDataSource
    _used: bool = PrivateAttr(False)

    def to_file(
        self,
        data_root: StrPath,
        data_name: str,
        bbox: Optional[Bbox] = None,
        time_tuple: Optional[TimeRange] = None,
        driver: Optional[str] = None,
        variables: Optional[Variables] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
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
            handle_nodata=handle_nodata,
            logger=logger,
            single_var_as_array=variables is None,
        )

        if obj is None:
            return None

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
        elif driver not in utils.GDAL_DRIVER_CODE_MAP.values():
            raise ValueError(f"RasterDataset: Driver {driver} unknown.")
        else:
            ext = utils.GDAL_EXT_CODE_MAP.get(driver)
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
        bbox: Optional[Bbox] = None,
        geom: Optional[Geom] = None,
        buffer: GeomBuffer = 0,
        zoom_level: Optional[int] = None,
        align: Optional[bool] = None,
        variables: Optional[Variables] = None,
        time_tuple: Optional[TimeRange] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        single_var_as_array: bool = True,
        cache_root: Optional[StrPath] = None,
        logger: Logger = logger,
    ):
        """Return a clipped, sliced and unified RasterDataset.

        For a detailed description see:
        :py:func:`~hydromt.data_catalog.DataCatalog.get_rasterdataset`
        """
        try:
            self._used = True  # mark used
            ds = self._read_data(
                geom,
                bbox,
                cache_root,
                zoom_level=zoom_level,
                logger=logger,
            )
            if has_no_data(ds):
                raise NoDataException()
            # rename variables and parse data and attrs
            ds = self._rename_vars(ds)
            ds = self._validate_spatial_dims(ds)
            ds = self._set_crs(ds, logger)
            ds = self._set_nodata(ds)
            ds = self._shift_time(ds, logger)
            # slice data
            ds = RasterDatasetAdapter._slice_data(
                ds,
                variables,
                geom,
                bbox,
                buffer,
                align,
                time_tuple,
                logger=logger,
            )
            if has_no_data(ds):
                raise NoDataException()

            # uniformize data
            ds = self._apply_unit_conversions(ds, logger)
            ds = self._set_metadata(ds)
            # return array if single var and single_var_as_array
            return self._single_var_as_array(ds, single_var_as_array, variables)
        except NoDataException:
            _exec_nodata_strat(
                f"No data was read from source: {self.source.name}",
                strategy=handle_nodata,
                logger=logger,
            )

    def _read_data(
        self,
        geom: Optional[Geom],
        bbox: Optional[Bbox],
        cache_root: Optional[StrPath],
        zoom_level: Optional[int] = None,
        logger: Logger = logger,
    ):
        logger.info(
            f"Reading {self.source.name} {self.source.driver} data from {self.source.uri}"
        )
        ds: xr.Dataset = self.source.read_data(
            mask=geom, bbox=bbox, zoom_level=zoom_level, **self.source.driver_kwargs
        )

        if has_no_data(ds):
            return None
        else:
            return ds

    def _rename_vars(self, ds: Data) -> Data:
        rm = {k: v for k, v in self.source.rename.items() if k in ds}
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

    def _set_crs(self, ds: Data, logger: Logger = logger) -> Data:
        # set crs
        if ds.raster.crs is None and self.source.crs is not None:
            ds.raster.set_crs(self.source.crs)
        elif ds.raster.crs is None:
            raise ValueError(
                f"RasterDataset {self.source.name}: CRS not defined in data catalog or data."
            )
        elif (
            self.source.crs is not None
            and ds.raster.crs != pyproj.CRS.from_user_input(self.source.crs)
        ):
            logger.warning(
                f"RasterDataset {self.source.name}: CRS from data catalog does not match CRS "
                " of data. The original CRS will be used. Please check your catalog."
            )
        return ds

    @staticmethod
    def _slice_data(
        ds: Data,
        variables: Optional[Variables] = None,
        geom: Optional[Geom] = None,
        bbox: Optional[Bbox] = None,
        buffer: GeomBuffer = 0,
        align: Optional[bool] = None,
        time_tuple: Optional[TimeRange] = None,
        logger: Logger = logger,
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
            ds = RasterDatasetAdapter._slice_temporal_dimension(
                ds,
                time_tuple,
                logger=logger,
            )
        if geom is not None or bbox is not None:
            ds = RasterDatasetAdapter._slice_spatial_dimensions(
                ds,
                geom,
                bbox,
                buffer,
                align,
                logger=logger,
            )

        if has_no_data(ds):
            return None
        else:
            return ds

    def _shift_time(self, ds: Data, logger: Logger = logger) -> Data:
        dt = self.source.unit_add.get("time", 0)
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
        ds: Data,
        time_tuple: Optional[TimeRange],
        logger: Logger = logger,
    ):
        if (
            "time" in ds.dims
            and ds["time"].size > 1
            and np.issubdtype(ds["time"].dtype, np.datetime64)
        ):
            if time_tuple is not None:
                logger.debug(f"Slicing time dim {time_tuple}")
                ds = ds.sel({"time": slice(*time_tuple)})
        if has_no_data(ds):
            return None
        else:
            return ds

    @staticmethod
    def _slice_spatial_dimensions(
        ds: Data,
        geom: Optional[Geom],
        bbox: Optional[Bbox],
        buffer: GeomBuffer,
        align: Optional[bool],
        logger: Logger = logger,
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
        w, _, e, _ = ds.raster.bounds
        if epsg == 4326 and np.isclose(e - w, 360):  # allow for rounding errors
            ds = utils.meridian_offset(ds, bbox)

        # clip with bbox
        if bbox is not None:
            bbox_str = ", ".join([f"{c:.3f}" for c in bbox])
            logger.debug(f"Clip to [{bbox_str}] (epsg:{epsg}))")
            ds = ds.raster.clip_bbox(bbox, buffer=buffer, align=align)
            # if np.any(np.array(ds.raster.shape) < 2):
            # check if bbox is fully covered
            w, s, e, n = ds.raster.bounds
            if not (w <= bbox[0] and s <= bbox[1] and e >= bbox[2] and n >= bbox[3]):
                logger.warning(
                    f"Dataset does [{w}, {s}, {e}, {n}] does not fully cover bbox [{bbox_str}]"
                )

        if has_no_data(ds):
            return None
        else:
            return ds

    def _apply_unit_conversions(self, ds: Data, logger=logger):
        unit_names = list(self.source.unit_mult.keys()) + list(
            self.source.unit_add.keys()
        )
        unit_names = [k for k in unit_names if k in ds.data_vars]
        if len(unit_names) > 0:
            logger.debug(f"Convert units for {len(unit_names)} variables.")
        for name in list(set(unit_names)):  # unique
            m = self.source.unit_mult.get(name, 1)
            a = self.source.unit_add.get(name, 0)
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
        if self.source.nodata is not None:
            if not isinstance(self.source.nodata, dict):
                nodata = {k: self.source.nodata for k in ds.data_vars.keys()}
            else:
                nodata = self.source.nodata
            for k in ds.data_vars:
                mv = nodata.get(k, None)
                if mv is not None and ds[k].raster.nodata is None:
                    ds[k].raster.set_nodata(mv)
        return ds

    def _set_metadata(self, ds):
        # unit attributes
        for k in self.source.attrs:
            ds[k].attrs.update(self.source.attrs[k])
        # set meta data
        ds.attrs.update(self.source.meta)
        return ds

    # TODO: uses rasterio and is specific to driver. Should be moved to driver
    def _get_zoom_levels_and_crs(
        self, fn: Optional[StrPath] = None, logger=logger
    ) -> Tuple[int, int]:
        """Get zoom levels and crs from adapter or detect from tif file if missing."""
        if self.source.zoom_levels is not None and self.source.crs is not None:
            return self.zoom_levels, self.source.crs
        zoom_levels = {}
        crs = None
        if fn is None:
            fn = self.source.uri
        try:
            with rasterio.open(fn) as src:
                res = abs(src.res[0])
                crs = src.crs
                overviews = [src.overviews(i) for i in src.indexes]
                if len(overviews[0]) > 0:  # check overviews for band 0
                    # check if identical
                    if not all([o == overviews[0] for o in overviews]):
                        raise ValueError("Overviews are not identical across bands")
                    # dict with overview level and corresponding resolution
                    zls = [1] + overviews[0]
                    zoom_levels = {i: res * zl for i, zl in enumerate(zls)}
        except RasterioIOError as e:
            logger.warning(f"IO error while detecting zoom levels: {e}")
        self.zoom_levels = zoom_levels
        if self.source.crs is None:
            self.source.crs = crs
        return zoom_levels, crs

    def _parse_zoom_level(
        self,
        zoom_level: Union[int, Tuple[Union[int, float], str]],
        geom: Optional[Geom] = None,
        bbox: Optional[Bbox] = None,
        zls_dict: Optional[Dict[int, float]] = None,
        dst_crs: Optional[pyproj.CRS] = None,
        logger=logger,
    ) -> Optional[int]:
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
        dst_crs = self.source.crs if dst_crs is None else dst_crs
        if zls_dict is None or len(zls_dict) == 0 or zoom_level is None:
            return None
        elif isinstance(zoom_level, int):
            if zoom_level not in zls_dict:
                raise ValueError(
                    f"Zoom level {zoom_level} not defined." f"Select from {zls_dict}."
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
                    conversions["degree"] = utils.cellres(lat=lat)[1]
                fsrc = conversions.get(src_res_unit, 1)
                fdst = conversions.get(dst_crs_unit, 1)
                dst_res = src_res * fsrc / fdst
            # find nearest zoom level
            res = list(zls_dict.values())[0] / 2
            zls = list(zls_dict.keys())
            smaller = [x < (dst_res + res * 0.01) for x in zls_dict.values()]
            zl = zls[-1] if all(smaller) else zls[max(smaller.index(False) - 1, 0)]
        elif dst_crs is None:
            raise ValueError("No CRS defined, hence no zoom level can be determined.")
        else:
            raise TypeError(f"zoom_level not understood: {type(zoom_level)}")
        logger.debug(f"Using zoom level {zl} ({dst_res:.2f})")
        return zl

    @staticmethod
    def _single_var_as_array(
        ds: Data, single_var_as_array: bool, variable_name: Optional[str] = None
    ) -> Data:
        # return data array if single variable dataset
        dvars = list(ds.data_vars.keys())
        if single_var_as_array and len(dvars) == 1:
            da = ds[dvars[0]]
            if isinstance(variable_name, list) and len(variable_name) == 1:
                da.name = variable_name[0]
            elif isinstance(variable_name, str):
                da.name = variable_name
            return da
        else:
            return ds

    def get_bbox(self, detect=True) -> TotalBounds:
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
        bbox = self.source.extent.get("bbox", None)
        crs = cast(int, self.source.crs)
        if bbox is None and detect:
            bbox, crs = self.detect_bbox()

        return bbox, crs

    def get_time_range(
        self,
        detect=True,
    ) -> TimeRange:
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
        time_range = self.source.extent.get("time_range", None)
        if time_range is None and detect:
            time_range = self.detect_time_range()

        return time_range

    def detect_bbox(
        self,
        ds=None,
    ) -> TotalBounds:
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

    def detect_time_range(
        self,
        ds=None,
    ) -> TimeRange:
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

    def to_stac_catalog(
        self,
        on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE,
    ) -> Optional[StacCatalog]:
        """
        Convert a rasterdataset into a STAC Catalog representation.

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
            bbox, crs = self.get_bbox(detect=True)
            bbox = list(bbox)
            start_dt, end_dt = self.get_time_range(detect=True)
            start_dt = pd.to_datetime(start_dt)
            end_dt = pd.to_datetime(end_dt)
            props = {**self.source.meta, "crs": crs}
            ext = splitext(self.source.uri)[-1]
            if ext == ".nc" or ext == ".vrt":
                media_type = MediaType.HDF5
            elif ext == ".tiff":
                media_type = MediaType.TIFF
            elif ext == ".cog":
                media_type = MediaType.COG
            elif ext == ".png":
                media_type = MediaType.PNG
            else:
                raise RuntimeError(
                    f"Unknown extention: {ext} cannot determine media type"
                )
        except (IndexError, KeyError, CRSError) as e:
            if on_error == ErrorHandleMethod.SKIP:
                logger.warning(
                    "Skipping {name} during stac conversion because"
                    "because detecting spacial extent failed."
                )
                return
            elif on_error == ErrorHandleMethod.COERCE:
                bbox = [0.0, 0.0, 0.0, 0.0]
                props = self.source.meta
                start_dt = datetime(1, 1, 1)
                end_dt = datetime(1, 1, 1)
                media_type = MediaType.JSON
            else:
                raise e

        else:
            # else makes type checkers a bit happier
            stac_catalog = StacCatalog(
                self.source.name,
                description=self.source.name,
            )
            stac_item = StacItem(
                self.source.name,
                geometry=None,
                bbox=list(bbox),
                properties=props,
                datetime=None,
                start_datetime=start_dt,
                end_datetime=end_dt,
            )
            stac_asset = StacAsset(str(self.source.uri), media_type=media_type)
            base_name = basename(self.source.uri)
            stac_item.add_asset(base_name, stac_asset)

            stac_catalog.add_item(stac_item)
            return stac_catalog
