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
from .raster import GEO_MAP_COORD
from .data_adapter import (
    DataAdapter, 
    RasterDatasetAdapter,
    GeoDatasetAdapter,
    GeoDataFrameAdapter,
)

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

