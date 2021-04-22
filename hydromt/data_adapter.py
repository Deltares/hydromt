#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""General data adapters for HydroMT"""

from abc import ABCMeta, abstractmethod, abstractproperty
import os
from os.path import join, isdir, dirname, basename, isfile, abspath
from itertools import product
import copy
from pathlib import Path
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import rasterio
import pyproj
import glob
import yaml
import pprint
import warnings
import logging
import requests
import shutil

from . import gis_utils, io

logger = logging.getLogger(__name__)

__all__ = [
    "DataCatalog",
]


class DataCatalog(object):
    # root URL and version with data source artifacts
    # url = f"{_url}/download/{_version}/<filename>"
    _url = r"https://github.com/DirkEilander/hydromt-artifacts/releases"
    _version = "v0.0.2"

    def __init__(self, data_libs=None, logger=logger, deltares_data=False):
        """Catalog of DataAdapter sources to easily read from different files
        and keep track of files which have been accessed.

        Arguments
        ---------
        data_libs: (list of) str, Path, optional
            One or more paths to yml files containing data sources which are parsed
            to entries of the data catalog. By default the data catalog is initiated
            without data entries. See :py:meth:`~hydromt.data_adapter.DataCatalog.from_yml`
            for accepted yml format.
        deltares_data: bool, optional
            If True run :py:meth:`~hydromt.data_adapter.DataCatalog.from_deltares_sources`
            to parse available Deltares global datasets library yml files.
        """
        self._sources = {}  # dictionary of DataAdapter
        self._used_data = []
        self.logger = logger
        if deltares_data:
            self.from_deltares_sources()
        if data_libs is not None:
            self.from_yml(data_libs)

    @property
    def sources(self):
        """Returns dictionary of DataAdapter sources."""
        if len(self._sources) == 0:
            self.from_artifacts()  # read artifacts by default
        return self._sources

    @property
    def keys(self):
        """Returns list of data source names."""
        return list(self.sources.keys())

    def __getitem__(self, key):
        return self.sources[key]

    def __setitem__(self, key, value):
        if not isinstance(value, DataAdapter):
            raise ValueError(f"Value must be DataAdapter, not {type(key).__name__}.")
        if key in self._sources:
            self.logger.warning(f"Overwriting data source {key}.")
        return self._sources.__setitem__(key, value)

    def __iter__(self):
        return self.sources.__iter__()

    def __len__(self):
        return self.sources.__len__()

    def __repr__(self):
        return self.to_dataframe().__repr__()

    def _repr_html_(self):
        return self.to_dataframe()._repr_html_()

    def update(self, **kwargs):
        """Add data sources to library."""
        for k, v in kwargs.items():
            self[k] = v

    def from_artifacts(self, version=None):
        """Add test data to data catalog.
        The data is available on https://github.com/DirkEilander/hydromt-artifacts and
        stored to to {user_home}/.hydromt/{version}/
        """
        # prepare url and paths
        version = self._version if version is None else version
        url = fr"{self._url}/download/{version}/data.tar.gz"
        folder = join(Path.home(), ".hydromt_data", "data", version)
        path_data = join(folder, "data.tar.gz")
        path_yml = join(folder, "data_catalog.yml")
        if not isdir(folder):
            os.makedirs(folder)
        # download data
        if not isfile(path_data):
            self.logger.info(f"Downloading file to {path_data}")
            with requests.get(url, stream=True) as r:
                with open(path_data, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
        if not isfile(path_yml):
            self.logger.debug(f"Unpacking data from {path_data}")
            shutil.unpack_archive(path_data, dirname(path_data))
        if not isfile(path_yml):
            raise FileNotFoundError(f"Data catalog file not found: {path_yml}")
        self.logger.info(f"Updating data sources from yml file {path_yml}")
        self.from_yml(path_yml)

    def to_yml(self, path, root=None):
        """Write data catalog to yml format.

        Parameters
        ----------
        path: str, Path
            yml oOutput path.
        root: str, Path, optional
            Global root for all relative paths in yml file.
            If None the data soruce paths are relative to the yml output ``path``.
        """
        if root is None:
            # set paths relative to yml file
            root = os.path.dirname(path)
            d = self.to_dict(root=root)
            d.pop("root", None)  # remoev absolute root path
        else:
            d = self.to_dict(root=root)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False)

    def from_yml(self, path, root=None):
        """Add data sources based on yml file.

        Parameters
        ----------
        path: iterable of str, Path
            Path(s) to data source yml files.
        root: str, Path, optional
            Global root for all relative paths in yml file(s).

        Examples
        --------
        A yml data entry is provided below, where all the text between <>
        should be filled by the user. Multiple data sources of the same
        data type should be grouped.  Currently the following data types are supported:
        {'RasterDataset', 'GeoDataset', 'GeoDataFrame'}. See the specific data adapters
        for more information about the required and optional arguments.

        .. code-block:: console

            root: <path>
            category: <path>
            <name>:
              path: <path>
              data_type: <data_type>
              driver: <driver>
              kwargs:
                <key>: <value>
              crs: <crs>
              nodata: <nodata>
              rename:
                <native_variable_name1>: <hydromt_variable_name1>
                <native_variable_name2>: <hydromt_variable_name2>
              unit_add:
                <native_variable_name1>: <float/int>
              unit_mult:
                <native_variable_name1>: <float/int>
              meta:
                source_url: <source_url>
                source_version: <source_version>
                source_licence: <source_licence>
                paper_ref: <paper_ref>
                paper_doi: <paper_doi>
        """
        self.update(**parse_data_sources(path=path, root=root))

    def from_dict(self, data_dict, root=None):
        """Add data sources based on dictionary.

        Parameters
        ----------
        data_dict: dict
            Dictionary of data_sources.
        root: str, Path, optional
            Global root for all relative paths in `data_dict`.

        Examples
        --------
        A data dictionary with two entries is provided below, where all the text between <>
        should be filled by the user. See the specific data adapters
        for more information about the required and optional arguments.

        .. code-block:: text

            {
                <name1>: {
                    "path": <path>,
                    "data_type": <data_type>,
                    "driver": <driver>,
                    "kwargs": {<key>: <value>},
                    "crs": <crs>,
                    "nodata": <nodata>,
                    "rename": {<native_variable_name1>: <hydromt_variable_name1>},
                    "unit_add": {<native_variable_name1>: <float/int>},
                    "unit_mult": {<native_variable_name1>: <float/int>},
                    "meta": {...}
                }
                <name2>: {
                    ...
                }
            }

        """
        self.update(**_parse_data_dict(data_dict, root=root))

    def from_deltares_sources(self, version=None):
        """Add global data sources from the Deltares network to the data catalog."""
        version = self._version if version is None else version
        url = rf"{self._url}/download/{version}/data_sources_deltares.yml"
        with requests.get(url, stream=True) as r:
            yml_str = r.text
        return self.from_dict(yaml.load(yml_str, Loader=yaml.FullLoader))

    def to_dict(self, source_names=[], root=None):
        """Return data catalog in dictionary format"""
        sources_out = dict()
        if root is not None:
            root = os.path.abspath(root)
            sources_out["root"] = root
            root_drive = os.path.splitdrive(root)[0]
        for name, source in self.sources.items():
            if len(source_names) > 0 and name not in source_names:
                continue
            source_dict = source.to_dict()
            if root is not None:
                path = source_dict["path"]  # is abspath
                source_drive = os.path.splitdrive(path)[0]
                if (
                    root_drive == source_drive
                    and os.path.commonpath([path, root]) == root
                ):
                    source_dict["path"] = os.path.relpath(source_dict["path"], root)
            sources_out.update({name: source_dict})
        return sources_out

    def to_dataframe(self, source_names=[]):
        """Return data catalog summary as DataFrame"""
        d = dict()
        for name, source in self.sources.items():
            if len(source_names) > 0 and name not in source_names:
                continue
            d[name] = source.summary()
        return pd.DataFrame.from_dict(d, orient="index")

    def export_data(
        self, data_root, bbox, time_tuple, source_names=[], unit_conversion=True
    ):
        """Export a data slice of each dataset and a data_catalog.yml file to disk.

        Parameters
        ----------
        data_root : str, Path
            Path to output folder
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        unit_conversion: boolean, optional
            If False skip unit conversion when parsing data from file, by default True.

        """
        if not os.path.isdir(data_root):
            os.makedirs(data_root)

        # create copy of data with selected source names
        sources = copy.deepcopy(self.sources)
        if len(source_names) > 0:
            sources = {n: sources[n] for n in source_names}

        # export data and update sources
        sources_out = {}
        for key, source in sources.items():
            try:
                # read slice of source and write to file
                self.logger.debug(f"Exporting {key}.")
                if not unit_conversion:
                    unit_mult = source.unit_mult
                    unit_add = source.unit_add
                    source.unit_mult = {}
                    source.unit_add = {}
                fn_out, driver = source.export_data(
                    data_root=data_root,
                    data_name=key,
                    bbox=bbox,
                    time_tuple=time_tuple,
                    logger=self.logger,
                )
                if fn_out is None:
                    self.logger.warning(f"{key} file contains no data within domain")
                    continue
                # update path & driver and remove kwargs and rename in output sources
                if unit_conversion:
                    source.unit_mult = {}
                    source.unit_add = {}
                else:
                    source.unit_mult = unit_mult
                    source.unit_add = unit_add
                source.path = fn_out
                source.driver = driver
                source.kwargs = {}
                source.rename = {}
                sources_out[key] = source
            except FileNotFoundError:
                self.logger.warning(f"{key} file not found at {source.path}")

        # write data catalog to yml
        data_catalog_out = DataCatalog()
        data_catalog_out._sources = sources_out
        fn = join(data_root, "data_catalog.yml")
        data_catalog_out.to_yml(fn)

    def get_rasterdataset(
        self,
        path_or_key,
        bbox=None,
        geom=None,
        buffer=0,
        align=None,
        variables=None,
        time_tuple=None,
        single_var_as_array=True,
        **kwargs,
    ):
        """Returns a clipped, sliced and unified RasterDataset from the data catalog.

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` and `align` arguments.
        To slice the data to the time period of interest, provide the `time_tuple` argument.
        To return only the dataset variables of interest and check their presence,
        provide the `variables` argument.

        NOTE: Unless `single_var_as_array` is set to False a single-varaible data source
        will be returned as xarray.DataArray rather than Dataset.

        Arguments
        ---------
        path_or_key: str
            Data catalog key. If a path to a raster file is provided it will be added
            to the data_catalog with its based on the file basename without extension.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest.
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        buffer : int, optional
            Buffer around the `bbox` or `geom` area of interest in pixels. By default 0.
        align : float, optional
            Resolution to align the bounding box, by default None
        variables : list of str, optional.
            Names of RasterDataset variables to return. By default all dataset variables
            are returned.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        single_var_as_array: bool, optional
            If True, return a DataArray if the dataset conists of a single variable.
            If False, always return a Dataset. By default True.

        Returns
        -------
        obj: xarray.Dataset or xarray.DataArray
            RasterDataset
        """
        if len(glob.glob(str(path_or_key))) > 0:
            path = path_or_key
            name = basename(path_or_key).split(".")[0]
            self.update(**{name: RasterDatasetAdapter(path=path, **kwargs)})
        elif path_or_key in self.sources:
            name = path_or_key
        else:
            raise FileNotFoundError(f"No such file or catalog key: {path_or_key}")
        self._used_data.append(name)
        source = self.sources[name]
        self.logger.info(
            f"DataCatalog: Getting {name} RasterDataset {source.driver} data from {source.path}"
        )
        obj = self.sources[name].get_data(
            bbox=bbox,
            geom=geom,
            buffer=buffer,
            align=align,
            variables=variables,
            time_tuple=time_tuple,
            single_var_as_array=single_var_as_array,
            logger=self.logger,
        )
        return obj

    def get_geodataframe(
        self,
        path_or_key,
        bbox=None,
        geom=None,
        buffer=0,
        variables=None,
        **kwargs,
    ):
        """Returns a clipped and unified GeoDataFrame (vector) from the data catalog.

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` and `align` arguments.
        To return only the dataframe columns of interest and check their presence,
        provide the `variables` argument.

        Arguments
        ---------
        path_or_key: str
            Data catalog key. If a path to a vector file is provided it will be added
            to the data_catalog with its based on the file basename without extension.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest.
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        buffer : float, optional
            Buffer around the `bbox` or `geom` area of interest in meters. By default 0.
        align : float, optional
            Resolution to align the bounding box, by default None
        variables : list of str, optional.
            Names of GeoDataFrame columns to return. By default all colums are returned.

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame
        """
        if isfile(path_or_key):
            path = path_or_key
            name = basename(path_or_key).split(".")[0]
            self.update(**{name: GeoDataFrameAdapter(path=path, **kwargs)})
        elif path_or_key in self.sources:
            name = path_or_key
        else:
            raise FileNotFoundError(f"No such file or catalog key: {path_or_key}")
        self._used_data.append(name)
        source = self.sources[name]
        self.logger.info(
            f"DataCatalog: Getting {name} GeoDataFrame {source.driver} data from {source.path}"
        )
        gdf = source.get_data(
            bbox=bbox,
            geom=geom,
            buffer=buffer,
            variables=variables,
            logger=self.logger,
        )
        return gdf

    def get_geodataset(
        self,
        path_or_key,
        bbox=None,
        geom=None,
        buffer=0,
        variables=None,
        time_tuple=None,
        single_var_as_array=True,
        **kwargs,
    ):
        """Returns a clipped, sliced and unified GeoDataset from the data catalog.

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` and `align` arguments.
        To slice the data to the time period of interest, provide the `time_tuple` argument.
        To return only the dataset variables of interest and check their presence,
        provide the `variables` argument.

        NOTE: Unless `single_var_as_array` is set to False a single-varaible data source
        will be returned as xarray.DataArray rather than Dataset.

        Arguments
        ---------
        path_or_key: str
            Data catalog key. If a path to a file is provided it will be added
            to the data_catalog with its based on the file basename without extension.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest.
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        buffer : float, optional
            Buffer around the `bbox` or `geom` area of interest in meters. By default 0.
        align : float, optional
            Resolution to align the bounding box, by default None
        variables : list of str, optional.
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        single_var_as_array: bool, optional
            If True, return a DataArray if the dataset conists of a single variable.
            If False, always return a Dataset. By default True.

        Returns
        -------
        obj: xarray.Dataset or xarray.DataArray
            GeoDataset
        """
        if isfile(path_or_key):
            path = path_or_key
            name = basename(path_or_key).split(".")[0]
            self.update(**{name: GeoDatasetAdapter(path=path, **kwargs)})
        elif path_or_key in self.sources:
            name = path_or_key
        else:
            raise FileNotFoundError(f"No such file or catalog key: {path_or_key}")
        self._used_data.append(name)
        source = self.sources[name]
        self.logger.info(
            f"DataCatalog: Getting {name} GeoDataset {source.driver} data from {source.path}"
        )
        obj = source.get_data(
            bbox=bbox,
            geom=geom,
            buffer=buffer,
            variables=variables,
            time_tuple=time_tuple,
            single_var_as_array=single_var_as_array,
            logger=self.logger,
        )
        return obj


def parse_data_sources(path=None, root=None):
    """Parse data sources yml file.
    For details see :py:meth:`~hydromt.data_adapter.DataCatalog.from_yml`
    """
    # check path argument
    if isinstance(path, (str, Path)):
        path = [path]
    elif not isinstance(path, (list, tuple)):
        raise ValueError(f"Unknown type for path argument: {type(path).__name__}")
    # check root argument
    if isinstance(root, (str, Path)):
        root = [root]
    elif root is not None and not isinstance(root, (list, tuple)):
        raise ValueError(f"Unknown type for root argument: {type(root).__name__}")
    if root is not None and len(root) == 1 and len(path) > 1:
        root = [root[0] for _ in range(len(path))]
    # loop over yml files
    data = dict()
    for i in range(len(path)):
        with open(path[i], "r") as stream:
            yml = yaml.load(stream, Loader=yaml.FullLoader)
        # read global root & category vars
        path0 = yml.pop("root", dirname(path[i]))
        path0 = path0 if root is None else root[i]
        category = yml.pop("category", None)
        data.update(**_parse_data_dict(yml, root=path0, category=category))
    return data


def _parse_data_dict(data_dict, root=None, category=None):
    """Parse data source dictionary."""
    # link yml keys to adapter classes
    ADAPTERS = {
        "RasterDataset": RasterDatasetAdapter,
        "GeoDataFrame": GeoDataFrameAdapter,
        "GeoDataset": GeoDatasetAdapter,
    }
    # set data_type sections as source entry
    # for backwards compatability
    sources = copy.deepcopy(data_dict)
    if root is None:
        root = sources.pop("root", None)
    for key in data_dict:
        _key = key.replace("Adapter", "")
        if _key in ADAPTERS:
            _sources = data_dict.pop(key)
            for name, source in _sources.items():
                source["data_type"] = _key
                sources[name] = source

    # parse data
    data = dict()
    for name, source in sources.items():
        if "path" not in source:
            raise ValueError("Missing required path argument.")
        data_type = source.pop("data_type", None)
        if data_type is None:
            raise ValueError("Data type missing.")
        elif data_type not in ADAPTERS:
            raise ValueError(f"Data type unknown: {data_type}")
        adapter = ADAPTERS.get(data_type)
        path = abs_path(root, source.pop("path"))
        meta = source.pop("meta", {})
        if "category" not in meta:
            meta.update(category=category)
        # lower kwargs for backwards compatability
        source.update(**source.pop("kwargs", {}))
        if "fn_ts" in source:
            source.update(fn_ts=abs_path(root, source["fn_ts"]))
        data[name] = adapter(path=path, meta=meta, **source)
    return data


def abs_path(root, rel_path):
    path = Path(str(rel_path))
    if not path.is_absolute():
        if root is not None:
            rel_path = join(root, rel_path)
        path = Path(abspath(rel_path))
    return str(path)


def round_latlon(ds, decimals=5):
    x_dim = ds.raster.x_dim
    y_dim = ds.raster.y_dim
    ds[x_dim] = np.round(ds[x_dim], decimals=decimals)
    ds[y_dim] = np.round(ds[y_dim], decimals=decimals)
    return ds


PREPROCESSORS = {"round_latlon": round_latlon}


class DataAdapter(object, metaclass=ABCMeta):
    """General Interface to data source for HydroMT"""

    _DEFAULT_DRIVER = None  # placeholder
    _DRIVERS = {}

    def __init__(
        self,
        path,
        driver,
        crs=None,
        nodata=None,
        rename={},
        unit_mult={},
        unit_add={},
        meta={},
        **kwargs,
    ):
        # general arguments
        self.path = path
        # driver and driver keyword-arguments
        # check for non default driver based on extension
        if driver is None:
            driver = self._DRIVERS.get(
                path.split(".")[-1].lower(), self._DEFAULT_DRIVER
            )
        self.driver = driver
        self.kwargs = kwargs
        # data adapter arguments
        self.crs = crs
        self.nodata = nodata
        self.rename = rename
        self.unit_mult = unit_mult
        self.unit_add = unit_add
        # meta data
        self.meta = meta

    @property
    def data_type(self):
        return type(self).__name__.replace("Adapter", "")

    def summary(self):
        """Returns a dictionary summary of the data adapter."""
        return dict(
            path=self.path,
            data_type=self.data_type,
            driver=self.driver,
            **self.meta,
        )

    def to_dict(self):
        """Returns a dictionary view of the data source. Can be used to initialize
        the data adapter."""
        source = dict(data_type=self.data_type)
        for k, v in vars(self).items():
            if v is not None and (not isinstance(v, dict) or len(v) > 0):
                source.update({k: v})
        return source

    def __str__(self):
        return pprint.pformat(self.summary())

    def __repr__(self):
        return self.__str__()

    def resolve_paths(self, time_tuple=None, variables=None):
        """Returns list of paths. Resolve {year}, {month} and {variable} keywords
        in self.path based 'time_tuple' and 'variables' arguments"""
        yr, mth = "*", "*"
        vrs = ["*"]
        dates = [""]
        fns = []
        if time_tuple is not None and "{year" in self.path:
            dt = pd.to_timedelta(self.unit_add.get("time", 0), unit="s")
            trange = pd.to_datetime(list(time_tuple)) - dt
            freq = "m" if "{month" in self.path else "a"
            dates = pd.period_range(*trange, freq=freq)
        if variables is not None and "{variable" in self.path:
            mv_inv = {v: k for k, v in self.rename.items()}
            vrs = [mv_inv.get(var, var) for var in variables]
        for date, var in product(dates, vrs):
            root, bname = dirname(self.path), basename(self.path)
            if hasattr(date, "month"):
                yr, mth = date.year, date.month
            bname = bname.format(year=yr, month=mth, variable=var)
            fns.extend(glob.glob(join(root, bname)))
        if len(fns) == 0:
            raise FileNotFoundError(f"No such file found: {self.path}")
        return list(set(fns))  # return unique paths

    @abstractmethod
    def get_data(self, bbox, geom, buffer):
        """Return a view (lazy if possible) of the data with standardized field names.
        If bbox of maks are given, clip data to that extent"""


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
        **kwargs,
    ):
        """Initiates data adapter for geospatial raster data.

        This object contains all properties required to read supported raster files into
        a single unified RasterDataset, i.e. :py:meth:`xarray.Dataset` with geospatial attributes.
        In addition it keeps meta data to be able to reproduce which data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source. If the dataset consists of multiple files, the path may
            contain {variable}, {year}, {month} placeholders as well as path search pattern
            using a '*' wildcard.
        driver: {'raster', 'netcdf', 'zarr'}, optional
            Driver to read files with, for 'raster' :py:meth:`~hydromt.io.open_mfraster`,
            for 'netcdf' :py:meth:`xarray.open_mfdataset`, and for 'zarr' :py:meth:`xarray.open_zarr`
            By default the driver is infered from the file extension and falls back to
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
            Metadata information of dataset, prefably containing the following keys:
            {'source_version', 'source_url', 'source_license', 'paper_ref', 'paper_doi', 'category'}
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
            **kwargs,
        )
        # TODO: see if the units argument can be solved with unit_mult/unit_add
        self.units = units

    def export_data(
        self,
        data_root,
        data_name,
        bbox,
        time_tuple,
        driver=None,
        variables=None,
        logger=logger,
        **kwargs,
    ):
        """Export a data slice to file.

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
            Name of driver to read data with, see :py:meth:`~hydromt.data_adapter.DataCatalog.get_rasterdataset`
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
            dvars = [obj.name] if isinstance(obj, xr.DataArray) else obj.raster.vars
            encoding = {k: {"zlib": True} for k in dvars}
            obj.to_netcdf(fn_out, encoding=encoding, **kwargs)
        elif driver == "zarr":
            fn_out = join(data_root, f"{data_name}.zarr")
            obj.to_zarr(fn_out, **kwargs)
        elif driver not in gis_utils.GDAL_DRIVER_CODE_MAP.values():
            raise ValueError(f"RasterDataset: Driver {driver} unknown.")
        else:
            ext = gis_utils.GDAL_EXT_CODE_MAP.get(driver)
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

        For a detailed description see: :py:meth:`~hydromt.data_adapter.DataCatalog.get_rasterdataset`
        """
        kwargs = self.kwargs.copy()
        fns = self.resolve_paths(time_tuple=time_tuple, variables=variables)

        # read using various readers
        if self.driver in ["netcdf"]:  # TODO complete list
            if "preprocess" in kwargs:
                preprocess = PREPROCESSORS.get(kwargs["preprocess"], None)
                kwargs.update(preprocess=preprocess)
            ds_out = xr.open_mfdataset(fns, **kwargs)
        elif self.driver == "zarr":
            if len(fns) > 1:
                raise ValueError(
                    "RasterDataset: Opening multiple zarr data files is not supported."
                )
            ds_out = xr.open_zarr(fns[0], **kwargs)
        elif self.driver == "raster_tindex":
            kwargs.update(nodata=self.nodata)
            ds_out = io.open_raster_from_tindex(fns[0], bbox=bbox, geom=geom, **kwargs)
        elif self.driver == "raster":  # rasterio files
            ds_out = io.open_mfraster(fns, logger=logger, **kwargs)
        else:
            raise ValueError(f"RasterDataset: Driver {self.driver} unknown")

        # rename and select vars
        if variables and len(ds_out.raster.vars) == 1 and len(self.rename) == 0:
            rm = {ds_out.raster.vars[0]: variables[0]}
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
        units={},
        meta={},
        **kwargs,
    ):
        """Initiates data adapter for geospatial timeseries data.

        This object contains all properties required to read supported files into
        a single unified GeoDataset, i.e. :py:meth:`xarray.Dataset` with geospatial point
        geometries. In addition it keeps meta data to be able to reproduce which data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source. If the dataset consists of multiple files, the path may
            contain {variable}, {year}, {month} placeholders as well as path search pattern
            using a '*' wildcard.
        driver: {'vector', 'netcdf'}, optional
            Driver to read files with, for 'vector' :py:meth:`~hydromt.io.open_geodataset`,
            for 'netcdf' :py:meth:`xarray.open_mfdataset`.
            By default the driver is infered from the file extension and falls back to
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
            **kwargs,
        )

    def export_data(
        self,
        data_root,
        data_name,
        bbox,
        time_tuple,
        variables=None,
        driver=None,
        logger=logger,
        **kwargs,
    ):
        """Export a data slice to file.

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
            Name of driver to read data with, see :py:meth:`~hydromt.data_adapter.DataCatalog.get_geodataset`
        """
        obj = self.get_data(
            bbox=bbox, time_tuple=time_tuple, variables=variables, logger=logger
        )
        if obj.vector.index.size == 0 or ("time" in obj and obj.time.size == 0):
            return None, None

        if driver is None or driver == "netcdf":
            # always write netcdf
            driver = "netcdf"
            fn_out = join(data_root, f"{data_name}.nc")
            dvars = [obj.name] if isinstance(obj, xr.DataArray) else obj.raster.vars
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

        For a detailed description see: :py:meth:`~hydromt.data_adapter.DataCatalog.get_geodataset`
        """
        kwargs = self.kwargs.copy()
        fns = self.resolve_paths(time_tuple=time_tuple, variables=variables)

        # parse geom, bbox and buffer arguments
        clip_str = ""
        if geom is None and bbox is not None:
            # convert bbox to geom with crs EPGS:4326 to apply buffer later
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
            bbox_str = ", ".join([f"{c:.3f}" for c in bbox])
            clip_str = f"and clip to bbox - [{bbox_str}]"
        elif geom is not None:
            bbox_str = ", ".join([f"{c:.3f}" for c in geom.total_bounds])
            clip_str = f"and clip to geom - [{bbox_str}]"
        if geom is not None:
            # make sure geom is projected > buffer in meters!
            if buffer > 0 and geom.crs.is_geographic:
                geom = geom.to_crs(3857)
            geom = geom.buffer(buffer)
        if kwargs.pop("within", False):  # for backward compatibility
            kwargs.update(predicate="contains")

        # read and clip
        ext = str(fns[0]).split(".")[-1].lower()
        logger.info(f"GeoDataset: Read {ext} data {clip_str}")
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
            ds_out = io.open_geodataset(fn_locs=fns[0], mask=geom, **kwargs)
            geom = None  # already clipped
        else:
            raise ValueError(f"GeoDataset: Driver {self.driver} unknown")

        # rename and select vars
        ds_out = ds_out.rename({k: v for k, v in self.rename.items() if k in ds_out})
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
                if mv is not None and ds_out[k].raster.nodata is None:
                    ds_out[k].raster.set_nodata(mv)

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
            nodata_isnan = da.raster.nodata is None or np.isnan(da.raster.nodata)
            # nodata value is explicitly set to NaN in case no nodata value is provided
            nodata = np.nan if nodata_isnan else da.raster.nodata
            data_bool = ~np.isnan(da) if nodata_isnan else da != nodata
            ds_out[name] = xr.where(data_bool, da * m + a, nodata)
            ds_out[name].attrs.update(attrs)  # set original attributes

        # return data array if single var
        if single_var_as_array and len(ds_out.raster.vars) == 1:
            ds_out = ds_out[ds_out.raster.vars[0]]

        # set meta data
        ds_out.attrs.update(self.meta)

        return ds_out


class GeoDataFrameAdapter(DataAdapter):
    _DEFAULT_DRIVER = "vector"
    _DRIVERS = {
        "xy": "xy",
        "csv": "csv",
        "xls": "xls",
        "xlsx": "xlsx",
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
        **kwargs,
    ):
        """Initiates data adapter for geospatial vector data.

        This object contains all properties required to read supported files into
        a single unified :py:meth:`geopandas.GeoDataFrame`.
        In addition it keeps meta data to be able to reproduce which data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source.
        driver: {'vector', 'xy', 'csv', 'xls', 'xlsx'}, optional
            Driver to read files with, for 'vector' :py:meth:`~geopandas.read_file`,
            for {'xy', 'csv', 'xls', 'xlsx'} :py:meth:`hydromt.io.open_vector_from_table`
            By default the driver is infered from the file extension and falls back to
            'vector' if unknown.
        crs: int, dict, or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str); proj (str or dict)
            or wkt (str). Only used if the data has no native CRS.
        nodata: (dictionary) float, int, optional
            Missing value number. Only used if the data has no native missing value.
            Multiple nodata values can be provided in a list and differentiated between
            dataframe columns using a dictionary with variable (column) keys. The nodata
            values are only applied to columns with numeric data.
        rename: dict, optional
            Mapping of native data source variable to output source variable name as
            required by hydroMT.
        unit_mult, unit_add: dict, optional
            Scaling multiplication and addition to change to map from the native data unit
            to the output data unit as required by hydroMT.
        meta: dict, optional
            Metadata information of dataset, prefably containing the following keys:
            {'source_version', 'source_url', 'source_license', 'paper_ref', 'paper_doi', 'category'}
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
            **kwargs,
        )

    def export_data(
        self,
        data_root,
        data_name,
        bbox,
        driver=None,
        variables=None,
        logger=logger,
        **kwargs,
    ):
        """Export a data slice to file.

        Parameters
        ----------
        data_root : str, Path
            Path to output folder
        data_name : str
            Name of output file without extension.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest.
        driver : str, optional
            Driver to write file, e.g.: 'GPKG', 'ESRI Shapefile' or any fiona data type, by default None
        variables : list of str, optional
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.

        Returns
        -------
        fn_out: str
            Absolute path to output file
        driver: str
            Name of driver to read data with, see :py:meth:`~hydromt.data_adapter.DataCatalog.get_geodataframe`
        """
        kwargs.pop("time_tuple", None)
        gdf = self.get_data(bbox=bbox, logger=logger)
        if gdf.index.size == 0:
            return None, None

        if driver is None:
            driver = "csv" if self.driver in ["csv", "xls", "xlsx", "xy"] else "GPKG"
        # always write netcdf
        if driver == "csv":
            fn_out = join(data_root, f"{data_name}.csv")
            if not np.all(gdf.geometry.type == "Point"):
                raise ValueError(
                    f"{data_name} contains other geometries than 'Point' "
                    "which cannot be written to csv."
                )
            gdf["x"], gdf["y"] = gdf.geometry.x, gdf.geometry.y
            gdf.drop(columns="geometry").to_csv(fn_out, **kwargs)
        else:
            driver_extensions = {
                "ESRI Shapefile": ".shp",
            }
            ext = driver_extensions.get(driver, driver).lower()
            fn_out = join(data_root, f"{data_name}.{ext}")
            gdf.to_file(fn_out, driver=driver, **kwargs)
            driver = "vector"

        return fn_out, driver

    def get_data(
        self,
        bbox=None,
        geom=None,
        predicate="intersects",
        buffer=0,
        logger=logger,
        variables=None,
        **kwargs,  # this is not used, for testing only
    ):
        """Returns a clipped and unified GeoDataFrame (vector) based on the properties
        of this GeoDataFrameAdapter.

        For a detailed description see: :py:meth:`~hydromt.data_adapter.DataCatalog.get_geodataframe`
        """
        kwargs = self.kwargs.copy()
        _ = self.resolve_paths()  # throw nice error if data not found

        # parse geom, bbox and buffer arguments
        clip_str = ""
        if geom is None and bbox is not None:
            # convert bbox to geom with crs EPGS:4326 to apply buffer later
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
            bbox_str = ", ".join([f"{c:.3f}" for c in bbox])
            clip_str = f"and clip to bbox - [{bbox_str}]"
        elif geom is not None:
            bbox_str = ", ".join([f"{c:.3f}" for c in geom.total_bounds])
            clip_str = f"and clip to geom - [{bbox_str}]"
        if geom is not None:
            # make sure geom is projected > buffer in meters!
            if geom.crs.is_geographic and buffer > 0:
                geom = geom.to_crs(3857)
            geom = geom.buffer(buffer)  # a buffer with zero fixes some topology errors
        if kwargs.pop("within", False):  # for backward compatibility
            kwargs.update(predicate="contains")

        # read and clip
        if self.driver in ["csv", "xls", "xlsx", "xy", "vector"]:
            gdf = io.open_vector(
                self.path, driver=self.driver, crs=self.crs, geom=geom, **kwargs
            )
        else:
            raise ValueError(f"GeoDataFrame: driver {self.driver} unknown.")

        # rename and select columns
        if self.rename:
            rename = {k: v for k, v in self.rename.items() if k in gdf.columns}
            gdf = gdf.rename(columns=rename)
        if variables is not None:
            if np.any([var not in gdf.columns for var in variables]):
                raise ValueError(f"GeoDataFrame: Not all variables found: {variables}")
            if "geometry" not in variables:  # always keep geometry column
                variables = variables + ["geometry"]
            gdf = gdf.loc[:, variables]

        # nodata and unit conversion for numeric data
        if gdf.index.size == 0:
            logger.warning(f"GeoDataFrame: No data within spatial domain {self.path}.")
        else:
            # parse nodata values
            cols = gdf.select_dtypes([np.number]).columns
            if self.nodata is not None and len(cols) > 0:
                if not isinstance(self.nodata, dict):
                    nodata = {c: self.nodata for c in cols}
                else:
                    nodata = self.nodata
                for c in cols:
                    mv = nodata.get(c, None)
                    if mv is not None:
                        is_nodata = np.isin(gdf[c], np.atleast_1d(mv))
                        gdf[c] = np.where(is_nodata, np.nan, gdf[c])

            # unit conversion
            unit_names = list(self.unit_mult.keys()) + list(self.unit_add.keys())
            unit_names = [k for k in unit_names if k in gdf.columns]
            if len(unit_names) > 0:
                logger.debug(
                    f"GeoDataFrame: Convert units for {len(unit_names)} columns."
                )
            for name in list(set(unit_names)):  # unique
                m = self.unit_mult.get(name, 1)
                a = self.unit_add.get(name, 0)
                gdf[name] = gdf[name] * m + a

        # set meta data
        gdf.attrs.update(self.meta)
        return gdf
