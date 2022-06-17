#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""General data adapters for HydroMT"""

from abc import ABCMeta, abstractmethod
import os
from os.path import join, isdir, dirname, basename, isfile, abspath, exists
from itertools import product
import copy
from pathlib import Path
from typing import Tuple
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import glob
import yaml
import pprint
import logging
import requests
from urllib.parse import urlparse
import shutil
from distutils.version import LooseVersion
import itertools
import warnings
from string import Formatter

from . import gis_utils, io

logger = logging.getLogger(__name__)

__all__ = [
    "DataCatalog",
]


class DataCatalog(object):
    # root URL and version with data artifacts
    # url = f"{_url}/download/{_version}/<filename>"
    _url = r"https://github.com/DirkEilander/hydromt-artifacts/releases"
    _version = "v0.0.6"  # latest version

    def __init__(self, data_libs=None, logger=logger, **artifact_keys):
        """Catalog of DataAdapter sources to easily read from different files
        and keep track of files which have been accessed.

        Arguments
        ---------
        data_libs: (list of) str, Path, optional
            One or more paths to yml files containing data sources which are parsed
            to entries of the data catalog. By default the data catalog is initiated
            without data entries. See :py:func:`~hydromt.data_adapter.DataCatalog.from_yml`
            for accepted yml format.
        artifact_keys:
            key-word arguments specifying the name and version of a hydroMT data artifact,
            to get the latest version use `True` instead of a version. For instance,
            to get the latest data catalog with Deltares Data use `deltares_data=True`;
            to get the latest

        """
        self._sources = {}  # dictionary of DataAdapter
        self._used_data = []
        self.logger = logger
        for name, version in artifact_keys.items():
            if version is None or not version:
                continue
            if isinstance(version, str) and LooseVersion(version) <= LooseVersion(
                "v0.0.4"
            ):
                raise ValueError("The minimal support version is v0.0.5")
            self.from_artifacts(name=name, version=version)
        if data_libs is not None:
            for path in np.atleast_1d(data_libs):
                self.from_yml(path)

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

    def from_artifacts(self, name=None, version=None):
        """Read a catalog file from https://github.com/DirkEilander/hydromt-artifacts releases.

        If no name is provided the artifact sample data is downloaded and
        stored to to {user_home}/.hydromt/{version}/

        Parameters
        ----------
        name: str, optional
            Catalog name. If None (default) sample data is downloaded.
        version: str, optional
            Release version. By default it takes the latest known release.
        """
        #
        version = version if isinstance(version, str) else self._version
        if name is None or name == "artifact_data":
            # prepare url and paths
            url = rf"{self._url}/download/{version}/data.tar.gz"
            folder = join(Path.home(), ".hydromt_data", "data", version)
            path_data = join(folder, "data.tar.gz")
            path = join(folder, "data_catalog.yml")
            if not isdir(folder):
                os.makedirs(folder)
            # download data
            if not isfile(path_data):
                with requests.get(url, stream=True) as r:
                    if r.status_code != 200:
                        self.logger.error(f"Artifact data {version} not found at {url}")
                        return
                    self.logger.info(f"Downloading file to {path_data}")
                    with open(path_data, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
            if not isfile(path):
                self.logger.debug(f"Unpacking data from {path_data}")
                shutil.unpack_archive(path_data, dirname(path_data))
            self.logger.info(f"Adding sample data {version} from artifacts")
        else:
            path = rf"{self._url}/download/{version}/{name}.yml"
            self.logger.info(f"Adding {name} {version} sources from {path}")
        self.from_yml(path)

    def from_yml(self, path, root=None, mark_used=False):
        """Add data sources based on yml file.

        Parameters
        ----------
        path: iterable of str, Path
            Path(s) to data source yml files.
        root: str, Path, optional
            Global root for all relative paths in yml file(s).
        mark_used: bool
            If True, append to used_data list.

        Examples
        --------
        A yml data entry is provided below, where all the text between <>
        should be filled by the user. Multiple data sources of the same
        data type should be grouped.  Currently the following data types are supported:
        {'RasterDataset', 'GeoDataset', 'GeoDataFrame'}. See the specific data adapters
        for more information about the required and optional arguments.

        .. code-block:: console

            root: <path>
            category: <category>
            <name>:
              path: <path>
              data_type: <data_type>
              driver: <driver>
              kwargs:
                <key>: <value>
              crs: <crs>
              nodata:
                <hydromt_variable_name1>: <nodata>
              rename:
                <native_variable_name1>: <hydromt_variable_name1>
                <native_variable_name2>: <hydromt_variable_name2>
              unit_add:
                <hydromt_variable_name1>: <float/int>
              unit_mult:
                <hydromt_variable_name1>: <float/int>
              meta:
                source_url: <source_url>
                source_version: <source_version>
                source_licence: <source_licence>
                paper_ref: <paper_ref>
                paper_doi: <paper_doi>
              placeholders:
                <placeholder_name_1>: <list of names>
                <placeholder_name_2>: <list of names>
        """
        if uri_validator(path):
            with requests.get(path, stream=True) as r:
                if r.status_code != 200:
                    raise IOError(f"URL {r.content}: {path}")
                yml = yaml.load(r.text, Loader=yaml.FullLoader)
        else:
            with open(path, "r") as stream:
                yml = yaml.load(stream, Loader=yaml.FullLoader)
        # parse data
        if root is None:
            root = yml.pop("root", dirname(path))
        self.from_dict(yml, root=root, mark_used=mark_used)

    def from_dict(self, data_dict, root=None, mark_used=False):
        """Add data sources based on dictionary.

        Parameters
        ----------
        data_dict: dict
            Dictionary of data_sources.
        root: str, Path, optional
            Global root for all relative paths in `data_dict`.
        mark_used: bool
            If True, append to used_data list.

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
                    "unit_add": {<hydromt_variable_name1>: <float/int>},
                    "unit_mult": {<hydromt_variable_name1>: <float/int>},
                    "meta": {...},
                    "placeholders": {<placeholder_name_1>: <list of names>},
                }
                <name2>: {
                    ...
                }
            }

        """
        category = data_dict.pop("category", None)
        data_dict = _parse_data_dict(data_dict, root=root, category=category)
        self.update(**data_dict)
        if mark_used:
            self._used_data.extend(list(data_dict.keys()))

    def to_yml(self, path, root="auto", source_names=[], used_only=False):
        """Write data catalog to yml format.

        Parameters
        ----------
        path: str, Path
            yml output path.
        root: str, Path, optional
            Global root for all relative paths in yml file.
            If "auto" the data source paths are relative to the yml output ``path``.
        source_names: list, optional
            List of source names to export; ignored if `used_only=True`
        used_only: bool
            If True, export only data entries kept in used_data list, by default False.
        """
        source_names = self._used_data if used_only else source_names
        yml_dir = os.path.dirname(path)
        if root == "auto":
            root = yml_dir
        d = self.to_dict(root=root, source_names=source_names)
        if str(root) == yml_dir:
            d.pop("root", None)  # remove root if it equals the yml_dir
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False)

    def to_dict(self, source_names=[], root=None):
        """Export the data catalog to a dictionary.

        Parameters
        ----------
        source_names : list, optional
            List of source names to export
        root : str, Path, optional
            Global root for all relative paths in yml file.

        Returns
        -------
        dict
            data catalog dictionary
        """
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
                    source_dict["path"] = os.path.relpath(
                        source_dict["path"], root
                    ).replace("\\", "/")
            # remove non serializable entries to prevent errors
            source_dict = _process_dict(source_dict, logger=self.logger)  # TODO TEST
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
        source_names: list, optional
            List of source names to export
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
            print(key)
            try:
                # read slice of source and write to file
                self.logger.debug(f"Exporting {key}.")
                if not unit_conversion:
                    unit_mult = source.unit_mult
                    unit_add = source.unit_add
                    source.unit_mult = {}
                    source.unit_add = {}
                fn_out, driver = source.to_file(
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
        data_catalog_out.to_yml(fn, root="auto")

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

        NOTE: Unless `single_var_as_array` is set to False a single-variable data source
        will be returned as :py:class:`xarray.DataArray` rather than :py:class:`xarray.Dataset`.

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
        variables : str or list of str, optional.
            Names of RasterDataset variables to return. By default all dataset variables
            are returned.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        single_var_as_array: bool, optional
            If True, return a DataArray if the dataset consists of a single variable.
            If False, always return a Dataset. By default True.

        Returns
        -------
        obj: xarray.Dataset or xarray.DataArray
            RasterDataset
        """
        if path_or_key not in self.sources and exists(abspath(path_or_key)):
            path = str(abspath(path_or_key))
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
        variables : str or list of str, optional.
            Names of GeoDataFrame columns to return. By default all columns are returned.

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame
        """
        if path_or_key not in self.sources and exists(abspath(path_or_key)):
            path = str(abspath(path_or_key))
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

        NOTE: Unless `single_var_as_array` is set to False a single-variable data source
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
        variables : str or list of str, optional.
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        single_var_as_array: bool, optional
            If True, return a DataArray if the dataset consists of a single variable.
            If False, always return a Dataset. By default True.

        Returns
        -------
        obj: xarray.Dataset or xarray.DataArray
            GeoDataset
        """
        if path_or_key not in self.sources and exists(abspath(path_or_key)):
            path = str(abspath(path_or_key))
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


def _parse_data_dict(data_dict, root=None, category=None):
    """Parse data source dictionary."""
    # link yml keys to adapter classes
    ADAPTERS = {
        "RasterDataset": RasterDatasetAdapter,
        "GeoDataFrame": GeoDataFrameAdapter,
        "GeoDataset": GeoDatasetAdapter,
    }
    # NOTE: shouldn't the kwarg overwrite the dict/yml ?
    if root is None:
        root = data_dict.pop("root", None)

    # parse data
    data = dict()
    for name, source in data_dict.items():
        source = source.copy()  # important as we modify with pop
        if "alias" in source:
            alias = source.pop("alias")
            if alias not in data_dict:
                raise ValueError(f"alias {alias} not found in data_dict.")
            # use alias source but overwrite any attributes with original source
            source_org = source.copy()
            source = data_dict[alias].copy()
            source.update(source_org)
        if "path" not in source:
            raise ValueError(f"{name}: Missing required path argument.")
        data_type = source.pop("data_type", None)
        if data_type is None:
            raise ValueError(f"{name}: Data type missing.")
        elif data_type not in ADAPTERS:
            raise ValueError(f"{name}: Data type {data_type} unknown")
        adapter = ADAPTERS.get(data_type)
        path = abs_path(root, source.pop("path"))
        meta = source.pop("meta", {})
        if "category" not in meta and category is not None:
            meta.update(category=category)
        # lower kwargs for backwards compatability
        # FIXME this could be problamatic if driver kwargs conflict DataAdapter arguments
        source.update(**source.pop("kwargs", {}))
        for opt in source:
            if "fn" in opt:  # get absolute paths for file names
                source.update({opt: abs_path(root, source[opt])})
        if "placeholders" in source:
            options = source["placeholders"]
            for combination in itertools.product(*options.values()):
                path_n = path
                name_n = name
                for k, v in zip(options.keys(), combination):
                    path_n = path_n.replace("{" + k + "}", v)
                    name_n = name_n.replace("{" + k + "}", v)
                data[name_n] = adapter(path=path_n, meta=meta, **source)
        else:
            data[name] = adapter(path=path, meta=meta, **source)

    return data


def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False


def _process_dict(d, logger=logger):
    """Recursively change dict values to keep only python literal structures."""
    for k, v in d.items():
        _check_key = isinstance(k, str)
        if _check_key and isinstance(v, dict):
            d[k] = _process_dict(v)
        elif _check_key and isinstance(v, Path):
            d[k] = str(v)  # path to string
        elif not _check_key or not isinstance(v, (list, str, int, float, bool)):
            d.pop(k)  # remove this entry
            logger.debug(f'Removing non-serializable entry "{k}"')
    return d


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


def to_datetimeindex(ds):
    if ds.indexes["time"].dtype == "O":
        ds["time"] = ds.indexes["time"].to_datetimeindex()
    return ds


def remove_duplicates(ds):
    return ds.sel(time=~ds.get_index("time").duplicated())


PREPROCESSORS = {
    "round_latlon": round_latlon,
    "to_datetimeindex": to_datetimeindex,
    "remove_duplicates": remove_duplicates,
}


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
        placeholders={},
        **kwargs,
    ):
        # general arguments
        self.path = path
        # driver and driver keyword-arguments
        # check for non default driver based on extension
        if driver is None:
            driver = self._DRIVERS.get(
                str(path).split(".")[-1].lower(), self._DEFAULT_DRIVER
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
        self.meta = {k: v for k, v in meta.items() if v is not None}

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
        return yaml.dump(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def resolve_paths(self, time_tuple: Tuple = None, variables: list = None):
        """Resolve {year}, {month} and {variable} keywords
        in self.path based on 'time_tuple' and 'variables' arguments

        Parameters
        ----------
        time_tuple : tuple of str, optional
            Start and end data in string format understood by :py:func:`pandas.to_timedelta`, by default None
        variables : list of str, optional
            List of variable names, by default None

        Returns
        -------
        List:
            list of filenames matching the path pattern given date range and variables
        """
        yr, mth = "*", "*"
        vrs = ["*"]
        dates = [""]
        fns = []
        path = str(self.path)
        known_keys = ["year", "month", "variable"]

        # Extract keys
        keys = [i[1] for i in Formatter().parse(path) if i[1] is not None]
        # Extract key formats
        format_keys = [i[2] for i in Formatter().parse(path) if i[1] is not None]
        # Store keys and keys format
        format_dict = dict(zip(keys, format_keys))
        for key in known_keys:
            if key not in format_dict:
                format_dict[key] = ""
        # Remove the format tags from the path
        while "" in format_keys:
            format_keys.remove("")
        for key in format_keys:
            path = path.replace(":" + key, "")

        # double unknown keys to escape these when formatting
        for key in [key for key in keys if key not in known_keys]:
            path = path.replace("{" + key + "}", "{{" + key + "}}")
        # resolve dates: month & year keys
        if time_tuple is not None and "year" in keys:
            dt = pd.to_timedelta(self.unit_add.get("time", 0), unit="s")
            trange = pd.to_datetime(list(time_tuple)) - dt
            freq = "m" if "month" in keys else "a"
            dates = pd.period_range(*trange, freq=freq)
        # resolve variables
        if variables is not None and "variable" in keys:
            mv_inv = {v: k for k, v in self.rename.items()}
            vrs = [mv_inv.get(var, var) for var in variables]
        for date, var in product(dates, vrs):
            if hasattr(date, "month"):
                yr, mth = date.year, date.month
            path1 = path.format(
                year="{:{}}".format(yr, format_dict["year"]),
                month="{:{}}".format(mth, format_dict["month"]),
                variable="{:{}}".format(var, format_dict["variable"]),
            )
            # FIXME: glob won't work with other than local file systems; use fsspec instead
            fns.extend(glob.glob(path1))
        if len(fns) == 0:
            raise FileNotFoundError(f"No such file found: {self.path}")
        return list(set(fns))  # return unique paths

    @abstractmethod
    def get_data(self, bbox, geom, buffer):
        """Return a view (lazy if possible) of the data with standardized field names.
        If bbox of mask are given, clip data to that extent"""


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
            Metadata information of dataset, prefably containing the following keys:
            {'source_version', 'source_url', 'source_license', 'paper_ref', 'paper_doi', 'category'}
        placeholders: dict, optional
            Placeholders to expand yml entry to multiple entries (name and path) based on placeholder values
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
        bbox,
        time_tuple,
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
            Name of driver to read data with, see :py:func:`~hydromt.data_adapter.DataCatalog.get_rasterdataset`
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

        For a detailed description see: :py:func:`~hydromt.data_adapter.DataCatalog.get_rasterdataset`
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
            Placeholders to expand yml entry to multiple entries (name and path) based on placeholder values
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
        bbox,
        time_tuple,
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
            Name of driver to read data with, see :py:func:`~hydromt.data_adapter.DataCatalog.get_geodataset`
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

        For a detailed description see: :py:func:`~hydromt.data_adapter.DataCatalog.get_geodataset`
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
        a single unified :py:func:`geopandas.GeoDataFrame`.
        In addition it keeps meta data to be able to reproduce which data is used.

        Parameters
        ----------
        path: str, Path
            Path to data source.
        driver: {'vector', 'vector_table'}, optional
            Driver to read files with, for 'vector' :py:func:`~geopandas.read_file`,
            for {'vector_table'} :py:func:`hydromt.io.open_vector_from_table`
            By default the driver is inferred from the file extension and falls back to
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

    def to_file(
        self,
        data_root,
        data_name,
        bbox,
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
            Name of driver to read data with, see :py:func:`~hydromt.data_adapter.DataCatalog.get_geodataframe`
        """
        kwargs.pop("time_tuple", None)
        gdf = self.get_data(bbox=bbox, variables=variables, logger=logger)
        if gdf.index.size == 0:
            return None, None

        if driver is None:
            _lst = ["csv", "xls", "xlsx", "xy", "vector_table"]
            driver = "csv" if self.driver in _lst else "GPKG"
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

        For a detailed description see: :py:func:`~hydromt.data_adapter.DataCatalog.get_geodataframe`
        """
        # If variable is string, convert to list
        if variables:
            variables = np.atleast_1d(variables).tolist()

        kwargs = self.kwargs.copy()
        _ = self.resolve_paths()  # throw nice error if data not found

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
            if geom.crs.is_geographic and buffer > 0:
                geom = geom.to_crs(3857)
            geom = geom.buffer(buffer)  # a buffer with zero fixes some topology errors
            bbox_str = ", ".join([f"{c:.3f}" for c in geom.total_bounds])
            clip_str = f"{clip_str} [{bbox_str}]"
        if kwargs.pop("within", False):  # for backward compatibility
            predicate = "contains"

        # read and clip
        logger.info(f"GeoDataFrame: Read {self.driver} data{clip_str}.")
        if self.driver in ["csv", "xls", "xlsx", "xy", "vector", "vector_table"]:
            # "csv", "xls", "xlsx", "xy" deprecated use vector_table instead.
            # specific driver should be added to open_vector kwargs
            if "driver" not in kwargs and self.driver in ["csv", "xls", "xlsx", "xy"]:
                kwargs.update(driver=self.driver)
            gdf = io.open_vector(
                self.path, crs=self.crs, geom=geom, predicate=predicate, **kwargs
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
