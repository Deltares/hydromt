"""Implementation for the geodataset DataAdapter."""
import logging
import warnings
from datetime import datetime
from os.path import basename, splitext
from typing import Optional, Union

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from pyproj.exceptions import CRSError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from pystac import MediaType

from hydromt.typing import ErrorHandleMethod, GeoDatasetSource, TimeRange, TotalBounds

from .. import gis_utils, io
from ..nodata import NoDataStrategy, _exec_nodata_strat
from ..raster import GEO_MAP_COORD
from .data_adapter import DataAdapter
from .utils import netcdf_writer, shift_dataset_time, zarr_writer

logger = logging.getLogger(__name__)

__all__ = ["GeoDatasetAdapter", "GeoDatasetSource"]


class GeoDatasetAdapter(DataAdapter):

    """DatasetAdapter for GeoDatasets."""

    _DEFAULT_DRIVER = "vector"
    _DRIVERS = {
        "nc": "netcdf",
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
        name: str = "",
        catalog_name: str = "",
        provider: Optional[str] = None,
        version: Optional[str] = None,
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
            search pattern using a ``*`` wildcard.
        driver: {'vector', 'netcdf', 'zarr'}, optional
            Driver to read files with,
            for 'vector' :py:func:`~hydromt.io.open_geodataset`,
            for 'netcdf' :py:func:`xarray.open_mfdataset`.
            By default the driver is inferred from the file extension and falls back to
            'vector' if unknown.
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
                "GeoDatasetAdapter driver is deprecated and will be removed "
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
        self.extent = extent

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
            fn_out = netcdf_writer(
                obj=obj, data_root=data_root, data_name=data_name, variables=variables
            )
        elif driver == "zarr":
            fn_out = zarr_writer(
                obj=obj, data_root=data_root, data_name=data_name, **kwargs
            )
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
        self.mark_as_used()  # mark used
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
        handle_nodata=NoDataStrategy.RAISE,
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
        handle_nodata : NoDataStrategy, optional
            How to handle no data values. By default NoDataStrategy.RAISE.
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
                ds, time_tuple, handle_nodata, logger=logger
            )
        if geom is not None or bbox is not None:
            ds = GeoDatasetAdapter._slice_spatial_dimension(
                ds, geom, bbox, buffer, predicate, handle_nodata, logger=logger
            )
        return ds

    @staticmethod
    def _slice_spatial_dimension(
        ds, geom, bbox, buffer, predicate, handle_nodata, logger=logger
    ):
        geom = gis_utils.parse_geom_bbox_buffer(geom, bbox, buffer)
        bbox_str = ", ".join([f"{c:.3f}" for c in geom.total_bounds])
        epsg = geom.crs.to_epsg()
        logger.debug(f"Clip {predicate} [{bbox_str}] (EPSG:{epsg})")
        ds = ds.vector.clip_geom(geom, predicate=predicate)
        if ds.vector.index.size == 0:
            _exec_nodata_strat("No data within spatial domain.", handle_nodata, logger)
        return ds

    def _shift_time(self, ds, logger=logger):
        dt = self.unit_add.get("time", 0)
        return shift_dataset_time(dt=dt, ds=ds, logger=logger)

    @staticmethod
    def _slice_temporal_dimension(
        ds, time_tuple, handle_nodata=NoDataStrategy.RAISE, logger=logger
    ):
        if (
            "time" in ds.dims
            and ds["time"].size > 1
            and np.issubdtype(ds["time"].dtype, np.datetime64)
        ):
            logger.debug(f"Slicing time dim {time_tuple}")
            ds = ds.sel(time=slice(*time_tuple))
            if ds.time.size == 0:
                _exec_nodata_strat(
                    "GeoDataset: Time slice out of range.", handle_nodata, logger=logger
                )
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

    def get_bbox(self, detect=True) -> TotalBounds:
        """Return the bounding box and espg code of the dataset.

        if the bounding box is not set and detect is True,
        :py:meth:`hydromt.GeoDatasetAdapter.detect_bbox` will be used to detect it.

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
        if bbox is None and detect:
            bbox, crs = self.detect_bbox()

        crs = self.crs

        return bbox, crs

    def get_time_range(self, detect=True) -> TimeRange:
        """Detect the time range of the dataset.

        if the time range is not set and detect is True,
        :py:meth:`hydromt.GeoDatasetAdapter.detect_time_range` will be used
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
    ) -> TotalBounds:
        """Detect the bounding box and crs of the dataset.

        If no dataset is provided, it will be fetched according to the settings in the
        adapter. also see :py:meth:`hydromt.GeoDatasetAdapter.get_data`. the
        coordinates are in the CRS of the dataset itself, which is also returned
        alongside the coordinates.


        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the bounding box of.
            If none is provided, :py:meth:`hydromt.GeoDatasetAdapter.get_data`
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

        crs = ds.vector.crs.to_epsg()
        bounds = ds.vector.bounds
        return bounds, crs

    def detect_time_range(self, ds=None) -> TimeRange:
        """Detect the temporal range of the dataset.

        If no dataset is provided, it will be fetched according to the settings in the
        adapter. also see :py:meth:`hydromt.GeoDatasetAdapter.get_data`.

        Parameters
        ----------
        ds: xr.Dataset, xr.DataArray, Optional
            the dataset to detect the time range of. It must have a time dimentsion set.
            If none is provided, :py:meth:`hydromt.GeoDatasetAdapter.get_data`
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
            ds[ds.vector.time_dim].min().values,
            ds[ds.vector.time_dim].max().values,
        )

    def to_stac_catalog(
        self,
        on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE,
    ) -> Optional[StacCatalog]:
        """
        Convert a geodataset into a STAC Catalog representation.

        The collection will contain an asset for each of the associated files.


        Parameters
        ----------
        - on_error (str, optional): The error handling strategy.
          Options are: "raise" to raise an error on failure, "skip" to skip
          the dataset on failure, and "coerce" (default) to set default
          values on failure.

        Returns
        -------
        - Optional[StacCatalog]: The STAC Catalog representation of the dataset, or
          None if the dataset was skipped.
        """
        try:
            bbox, crs = self.get_bbox(detect=True)
            bbox = list(bbox)
            start_dt, end_dt = self.get_time_range(detect=True)
            start_dt = pd.to_datetime(start_dt)
            end_dt = pd.to_datetime(end_dt)
            props = {**self.meta, "crs": crs}
            ext = splitext(self.path)[-1]
            if ext in [".nc", ".vrt"]:
                media_type = MediaType.HDF5
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
                props = self.meta
                start_dt = datetime(1, 1, 1)
                end_dt = datetime(1, 1, 1)
                media_type = MediaType.JSON
            else:
                raise e

        stac_catalog = StacCatalog(
            self.name,
            description=self.name,
        )
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
