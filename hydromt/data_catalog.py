#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DataCatalog module for HydroMT"""
from __future__ import annotations
import os
from os.path import join, isdir, dirname, basename, isfile, abspath, exists
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import yaml
import logging
import requests
import shutil
from packaging.version import Version
import itertools

from .data_adapter import (
    DataAdapter,
    RasterDatasetAdapter,
    GeoDatasetAdapter,
    GeoDataFrameAdapter,
    DataFrameAdapter,
)
from .data_adapter.caching import _uri_validator, _copyfile, HYDROMT_DATADIR

logger = logging.getLogger(__name__)

__all__ = [
    "DataCatalog",
]


class DataCatalog(object):
    # root URL with data_catalog file
    _url = r"https://raw.githubusercontent.com/Deltares/hydromt/main/data/predefined_catalogs.yml"
    _cache_dir = HYDROMT_DATADIR

    def __init__(
        self,
        data_libs: Union[List, str] = [],
        fallback_lib: Optional[str] = "artifact_data",
        logger=logger,
        cache: bool = False,
        cache_dir: str = None,
        **artifact_keys,
    ) -> None:
        """Catalog of DataAdapter sources to easily read from different files
        and keep track of files which have been accessed.

        Arguments
        ---------
        data_libs: (list of) str, Path, optional
            One or more paths to data catalog yaml files or names of predefined data catalogs.
            By default the data catalog is initiated without data entries.
            See :py:func:`~hydromt.data_adapter.DataCatalog.from_yml` for accepted yaml format.
        fallback_lib:
            Name of pre-defined data catalog to read if no data_libs are provided, by default 'artifact_data'.
            If None, no default data catalog is used.
        cache: bool, optional
            Set to true to cache data locally before reading.
            Currently only implemented for tiled rasterdatasets, by default False.
        cache_dir: str, Path, optional
            Folder root path to cach data to, by default ~/.hydromt_data
        artifact_keys:
            Deprecated from version v0.5
        """
        if data_libs is None:  # legacy code. to be removed
            data_libs = []
        elif not isinstance(data_libs, list):  # make sure data_libs is a list
            data_libs = np.atleast_1d(data_libs).tolist()
        self._sources = {}  # dictionary of DataAdapter
        self._catalogs = {}  # dictionary of predefined Catalogs
        self._used_data = []
        self._fallback_lib = fallback_lib
        self.logger = logger

        # caching
        self.cache = bool(cache)
        if cache_dir is not None:
            self._cache_dir = cache_dir

        # legacy code. to be removed
        for lib, version in artifact_keys.items():
            warnings.warn(
                "Adding a predefined data catalog as key-word argument is deprecated, "
                f"add the catalog as '{lib}={version}' to the data_libs list instead.",
                DeprecationWarning,
            )
            if not version:  # False or None
                continue
            elif isinstance(version, str):
                lib += f"={version}"
            data_libs = [lib] + data_libs

        # parse data catalogs; both user and pre-defined
        for name_or_path in data_libs:
            if str(name_or_path).split(".")[-1] in ["yml", "yaml"]:  # user defined
                self.from_yml(name_or_path)
            else:  # predefined
                self.from_predefined_catalogs(name_or_path)

    @property
    def sources(self) -> Dict:
        """Returns dictionary of DataAdapter sources."""
        if len(self._sources) == 0 and self._fallback_lib is not None:
            # read artifacts by default if no catalogs are provided
            self.from_predefined_catalogs(self._fallback_lib)
        return self._sources

    @property
    def keys(self) -> List:
        """Returns list of data source names."""
        return list(self._sources.keys())

    @property
    def predefined_catalogs(self) -> Dict:
        if not self._catalogs:
            self.set_predefined_catalogs()
        return self._catalogs

    def __getitem__(self, key: str) -> DataAdapter:
        return self._sources[key]

    def __setitem__(self, key: str, value: DataAdapter) -> None:
        if not isinstance(value, DataAdapter):
            raise ValueError(f"Value must be DataAdapter, not {type(key).__name__}.")
        if key in self._sources:
            self.logger.warning(f"Overwriting data source {key}.")
        return self._sources.__setitem__(key, value)

    def __iter__(self):
        return self._sources.__iter__()

    def __len__(self):
        return self._sources.__len__()

    def __repr__(self):
        return self.to_dataframe().__repr__()

    def _repr_html_(self):
        return self.to_dataframe()._repr_html_()

    def update(self, **kwargs) -> None:
        """Add data sources to library."""
        for k, v in kwargs.items():
            self[k] = v

    def set_predefined_catalogs(self, urlpath: Union[Path, str] = None) -> Dict:
        # get predefined_catalogs
        urlpath = self._url if urlpath is None else urlpath
        # TODO cache predefined catalogs
        self._catalogs = _yml_from_uri_or_path(urlpath)
        return self._catalogs

    def from_artifacts(
        self, name: str = "artifact_data", version: str = "latest"
    ) -> None:
        """Deprecated method. Use :py:func:`hydromt.data_catalog.DataCatalog.from_predefined_catalogs` instead

        Parameters
        ----------
        name : str, optional
            Catalog name. If None (default) sample data is downloaded.
        version : str, optional
            Release version. By default it takes the latest known release.
        """
        warnings.warn(
            f'"from_artifacts" is deprecated. Use "from_predefined_catalogs instead".',
            DeprecationWarning,
        )
        self.from_predefined_catalogs(name, version)

    def from_predefined_catalogs(self, name: str, version: str = "latest") -> None:
        if "=" in name:
            name, version = name.split("=")[0], name.split("=")[-1]
        if name not in self.predefined_catalogs:
            raise ValueError(
                f'Catalog with name "{name}" not found in predefined catalogs'
            )
        urlpath = self.predefined_catalogs[name].get("urlpath")
        versions_dict = self.predefined_catalogs[name].get("versions")
        if version == "latest" or not isinstance(version, str):
            versions = list(versions_dict.keys())
            if len(versions) > 1:
                version = versions[np.argmax([Version(v) for v in versions])]
            else:
                version = versions[0]
        urlpath = urlpath.format(version=versions_dict.get(version, version))
        if urlpath.split(".")[-1] in ["gz", "zip"]:
            self.logger.info(f"Reading data catalog {name} {version} from archive")
            self.from_archive(urlpath, name=name, version=version)
        else:
            self.logger.info(f"Reading data catalog {name} {version}")
            self.from_yml(urlpath)

    def from_archive(
        self, urlpath: Union[Path, str], version: str = None, name: str = None
    ) -> None:
        """Read a data archive including a data_catalog.yml file"""
        name = basename(urlpath).split(".")[0] if name is None else name
        root = join(self._cache_dir, name)
        if version is not None:
            root = join(root, version)
        archive_fn = join(root, basename(urlpath))
        yml_fn = join(root, "data_catalog.yml")
        if not isdir(root):
            os.makedirs(root)
        # download data if url
        if _uri_validator(str(urlpath)) and not isfile(archive_fn):
            _copyfile(urlpath, archive_fn)
        # unpack data
        if not isfile(yml_fn):
            self.logger.debug(f"Unpacking data from {archive_fn}")
            shutil.unpack_archive(archive_fn, root)
        # parse catalog
        self.from_yml(yml_fn)

    def from_yml(
        self, urlpath: Union[Path, str], root: str = None, mark_used: bool = False
    ) -> None:
        """Add data sources based on yaml file.

        Parameters
        ----------
        urlpath: str, Path
            Path or url to data source yaml files.
        root: str, Path, optional
            Global root for all relative paths in yaml file(s).
        mark_used: bool
            If True, append to used_data list.

        Examples
        --------
        A yaml data entry is provided below, where all the text between <>
        should be filled by the user. Multiple data sources of the same
        data type should be grouped.  Currently the following data types are supported:
        {'RasterDataset', 'GeoDataset', 'GeoDataFrame'}. See the specific data adapters
        for more information about the required and optional arguments.

        .. code-block:: console

            meta:
              root: <path>
              category: <category>
              version: <version>
            <name>:
              path: <path>
              data_type: <data_type>
              driver: <driver>
              filesystem: <filesystem>
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
              zoom_levels:
                <zoom_level_1>: <resolution_1>
                <zoom_level_2>: <resolution_2>
        """
        self.logger.info(f"Parsing data catalog from {urlpath}")
        yml = _yml_from_uri_or_path(urlpath)
        # parse metadata
        meta = dict()
        # legacy code with root/category at highest yml level
        if "root" in yml:
            meta.update(root=yml.pop("root"))
        if "category" in yml:
            meta.update(category=yml.pop("category"))
        # read meta data
        meta = yml.pop("meta", meta)
        catalog_name = meta.get("name", "".join(basename(urlpath).split(".")[:-1]))
        # TODO keep meta data!! Note only possible if yml files are not merged
        if root is None:
            root = meta.get("root", os.path.dirname(urlpath))
        self.from_dict(
            yml,
            catalog_name=catalog_name,
            root=root,
            category=meta.get("category", None),
            mark_used=mark_used,
        )

    def from_dict(
        self,
        data_dict: Dict,
        catalog_name: str = "",
        root: Union[str, Path] = None,
        category: str = None,
        mark_used: bool = False,
    ) -> None:
        """Add data sources based on dictionary.

        Parameters
        ----------
        data_dict: dict
            Dictionary of data_sources.
        catalog_name: str, optional
            Name of data catalog
        root: str, Path, optional
            Global root for all relative paths in `data_dict`.
        category: str, optional
            Global category for all sources in `data_dict`.
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
                    "filesystem": <filesystem>,
                    "kwargs": {<key>: <value>},
                    "crs": <crs>,
                    "nodata": <nodata>,
                    "rename": {<native_variable_name1>: <hydromt_variable_name1>},
                    "unit_add": {<hydromt_variable_name1>: <float/int>},
                    "unit_mult": {<hydromt_variable_name1>: <float/int>},
                    "meta": {...},
                    "placeholders": {<placeholder_name_1>: <list of names>},
                    "zoom_levels": {<zoom_level_1>: <resolution_1>},
                }
                <name2>: {
                    ...
                }
            }

        """
        data_dict = _parse_data_dict(
            data_dict,
            catalog_name=catalog_name,
            root=root,
            category=category,
        )
        self.update(**data_dict)
        if mark_used:
            self._used_data.extend(list(data_dict.keys()))

    def to_yml(
        self,
        path: Union[str, Path],
        root: str = "auto",
        source_names: Optional[List] = None,
        used_only: bool = False,
        meta: Dict = {},
    ) -> None:
        """Write data catalog to yaml format.

        Parameters
        ----------
        path: str, Path
            yaml output path.
        root: str, Path, optional
            Global root for all relative paths in yaml file.
            If "auto" (default) the data source paths are relative to the yaml output ``path``.
        source_names: list, optional
            List of source names to export, by default None in which case all sources are exported.
            This argument is ignored if `used_only=True`.
        used_only: bool, optional
            If True, export only data entries kept in used_data list, by default False.
        meta: dict, optional
            key-value pairs to add to the data catalog meta section, such as 'version', by default empty.
        """
        source_names = self._used_data if used_only else source_names
        yml_dir = os.path.dirname(abspath(path))
        if root == "auto":
            root = yml_dir
        data_dict = self.to_dict(root=root, source_names=source_names, meta=meta)
        if str(root) == yml_dir:
            data_dict.pop("root", None)  # remove root if it equals the yml_dir
        if data_dict:
            with open(path, "w") as f:
                yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)
        else:
            self.logger.info("The data catalog is empty, no yml file is written.")

    def to_dict(
        self,
        source_names: Optional[List] = None,
        root: Union[Path, str] = None,
        meta: dict = {},
    ) -> Dict:
        """Export the data catalog to a dictionary.

        Parameters
        ----------
        source_names : list, optional
            List of source names to export, by default None in which case all sources are exported.
        root : str, Path, optional
            Global root for all relative paths in yml file.
        meta: dict, optional
            key-value pairs to add to the data catalog meta section, such as 'version', by default empty.

        Returns
        -------
        dict
            data catalog dictionary
        """
        sources_out = dict()
        if root is not None:
            root = abspath(root)
            meta.update(**{"root": root})
            root_drive = os.path.splitdrive(root)[0]
        for name, source in sorted(self._sources.items()):  # alphabetical order
            if source_names is not None and name not in source_names:
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
        if meta:
            sources_out = {"meta": meta, **sources_out}
        return sources_out

    def to_dataframe(self, source_names: List = []) -> pd.DataFrame:
        """Return data catalog summary as DataFrame"""
        d = dict()
        for name, source in self._sources.items():
            if len(source_names) > 0 and name not in source_names:
                continue
            d[name] = source.summary()
        return pd.DataFrame.from_dict(d, orient="index")

    def export_data(
        self,
        data_root: Union[Path, str],
        bbox: List = None,
        time_tuple: Tuple = None,
        source_names: List = [],
        unit_conversion: bool = True,
        meta: Dict = {},
    ) -> None:
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
        meta: dict, optional
            key-value pairs to add to the data catalog meta section, such as 'version', by default empty.
        """
        data_root = abspath(data_root)
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
                source.filesystem = "local"
                source.kwargs = {}
                source.rename = {}
                sources_out[key] = source
            except FileNotFoundError:
                self.logger.warning(f"{key} file not found at {source.path}")

        # write data catalog to yml
        data_catalog_out = DataCatalog()
        data_catalog_out._sources = sources_out
        fn = join(data_root, "data_catalog.yml")
        data_catalog_out.to_yml(fn, root="auto", meta=meta)

    def get_rasterdataset(
        self,
        data_like: Union[str, Path, xr.Dataset, xr.DataArray],
        bbox: List = None,
        geom: gpd.GeoDataFrame = None,
        zoom_level: int | tuple = None,
        buffer: Union[float, int] = 0,
        align: bool = None,
        variables: Union[List, str] = None,
        time_tuple: Tuple = None,
        single_var_as_array: bool = True,
        **kwargs,
    ) -> xr.Dataset:
        """Returns a clipped, sliced and unified RasterDataset.

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` and `align` arguments.
        To slice the data to the time period of interest, provide the `time_tuple` argument.
        To return only the dataset variables of interest provide the `variables` argument.

        NOTE: Unless `single_var_as_array` is set to False a single-variable data source
        will be returned as :py:class:`xarray.DataArray` rather than :py:class:`xarray.Dataset`.

        Arguments
        ---------
        data_like: str, Path, xr.Dataset, xr.Datarray
            Data catalog key, path to raster file or raster xarray data object.
            If a path to a raster file is provided it will be added
            to the data_catalog with its based on the file basename without extension.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest (in WGS84 coordinates).
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        zoom_level : int, tuple, optional
            Zoom level of the xyz tile dataset (0 is base level)
            Using a tuple the zoom level can be specified as (<zoom_resolution>, <unit>),
            e.g., (1000, 'meter')
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
        if isinstance(data_like, (xr.DataArray, xr.Dataset)):
            return data_like
        elif not isinstance(data_like, (str, Path)):
            raise ValueError(f'Unknown raster data type "{type(data_like).__name__}"')

        if data_like not in self.sources and exists(abspath(data_like)):
            path = str(abspath(data_like))
            name = basename(data_like).split(".")[0]
            self.update(**{name: RasterDatasetAdapter(path=path, **kwargs)})
        elif data_like in self.sources:
            name = data_like
        else:
            raise FileNotFoundError(f"No such file or catalog key: {data_like}")

        self._used_data.append(name)
        source = self.sources[name]
        self.logger.info(
            f"DataCatalog: Getting {name} RasterDataset {source.driver} data from {source.path}"
        )
        obj = self.sources[name].get_data(
            bbox=bbox,
            geom=geom,
            buffer=buffer,
            zoom_level=zoom_level,
            align=align,
            variables=variables,
            time_tuple=time_tuple,
            single_var_as_array=single_var_as_array,
            cache_root=self._cache_dir if self.cache else None,
            logger=self.logger,
        )
        return obj

    def get_geodataframe(
        self,
        data_like: Union[str, Path, gpd.GeoDataFrame],
        bbox: List = None,
        geom: gpd.GeoDataFrame = None,
        buffer: Union[float, int] = 0,
        variables: Union[List, str] = None,
        predicate: str = "intersects",
        **kwargs,
    ):
        """Returns a clipped and unified GeoDataFrame (vector).

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` and `align` arguments.
        To return only the dataframe columns of interest provide the `variables` argument.

        Arguments
        ---------
        data_like: str, Path, gpd.GeoDataFrame
            Data catalog key, path to vector file or a vector geopandas object.
            If a path to a vector file is provided it will be added
            to the data_catalog with its based on the file basename without extension.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest (in WGS84 coordinates).
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        buffer : float, optional
            Buffer around the `bbox` or `geom` area of interest in meters. By default 0.
        predicate : {'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches'}, optional
            If predicate is provided, the GeoDataFrame is filtered by testing
            the predicate function against each item. Requires bbox or mask.
            By default 'intersects'
        align : float, optional
            Resolution to align the bounding box, by default None
        variables : str or list of str, optional.
            Names of GeoDataFrame columns to return. By default all columns are returned.

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame
        """
        if isinstance(data_like, gpd.GeoDataFrame):
            return data_like
        elif not isinstance(data_like, (str, Path)):
            raise ValueError(f'Unknown vector data type "{type(data_like).__name__}"')

        if data_like not in self.sources and exists(abspath(data_like)):
            path = str(abspath(data_like))
            name = basename(data_like).split(".")[0]
            self.update(**{name: GeoDataFrameAdapter(path=path, **kwargs)})
        elif data_like in self.sources:
            name = data_like
        else:
            raise FileNotFoundError(f"No such file or catalog key: {data_like}")

        self._used_data.append(name)
        source = self.sources[name]
        self.logger.info(
            f"DataCatalog: Getting {name} GeoDataFrame {source.driver} data from {source.path}"
        )
        gdf = source.get_data(
            bbox=bbox,
            geom=geom,
            buffer=buffer,
            predicate=predicate,
            variables=variables,
            logger=self.logger,
        )
        return gdf

    def get_geodataset(
        self,
        data_like: Union[Path, str, xr.DataArray, xr.Dataset],
        bbox: List = None,
        geom: gpd.GeoDataFrame = None,
        buffer: Union[float, int] = 0,
        variables: List = None,
        time_tuple: Tuple = None,
        single_var_as_array: bool = True,
        **kwargs,
    ) -> xr.Dataset:
        """Returns a clipped, sliced and unified GeoDataset.

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` and `align` arguments.
        To slice the data to the time period of interest, provide the `time_tuple` argument.
        To return only the dataset variables of interest provide the `variables` argument.

        NOTE: Unless `single_var_as_array` is set to False a single-variable data source
        will be returned as xarray.DataArray rather than Dataset.

        Arguments
        ---------
        data_like: str, Path, xr.Dataset, xr.DataArray
            Data catalog key, path to geodataset file or geodataset xarray object.
            If a path to a file is provided it will be added
            to the data_catalog with its based on the file basename without extension.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest (in WGS84 coordinates).
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
        if isinstance(data_like, (xr.DataArray, xr.Dataset)):
            return data_like
        elif not isinstance(data_like, (str, Path)):
            raise ValueError(f'Unknown geo data type "{type(data_like).__name__}"')

        if data_like not in self.sources and exists(abspath(data_like)):
            path = str(abspath(data_like))
            name = basename(data_like).split(".")[0]
            self.update(**{name: GeoDatasetAdapter(path=path, **kwargs)})
        elif data_like in self.sources:
            name = data_like
        else:
            raise FileNotFoundError(f"No such file or catalog key: {data_like}")

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

    def get_dataframe(
        self,
        data_like: Union[str, Path, pd.DataFrame],
        variables: list = None,
        time_tuple: tuple = None,
        **kwargs,
    ):
        """Returns a unified and sliced DataFrame.

        Parameters
        ----------
        data_like : str, Path, pd.DataFrame
            Data catalog key, path to tabular data file or tabular pandas dataframe object.
            If a path to a tabular data file is provided it will be added
            to the data_catalog with its based on the file basename without extension.
        variables : str or list of str, optional.
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.

        Returns
        -------
        pd.DataFrame
            Tabular data
        """
        if isinstance(data_like, pd.DataFrame):
            return data_like
        elif not isinstance(data_like, (str, Path)):
            raise ValueError(f'Unknown tabular data type "{type(data_like).__name__}"')

        if data_like not in self.sources and exists(abspath(data_like)):
            path = str(abspath(data_like))
            name = basename(data_like).split(".")[0]
            self.update(**{name: DataFrameAdapter(path=path, **kwargs)})
        elif data_like in self.sources:
            name = data_like
        else:
            raise FileNotFoundError(f"No such file or catalog key: {data_like}")

        self._used_data.append(name)
        source = self.sources[name]
        self.logger.info(
            f"DataCatalog: Getting {name} DataFrame {source.driver} data from {source.path}"
        )
        obj = source.get_data(
            variables=variables,
            time_tuple=time_tuple,
            logger=self.logger,
        )
        return obj


def _parse_data_dict(
    data_dict: Dict,
    catalog_name: str = "",
    root: Union[Path, str] = None,
    category: str = None,
) -> Dict:
    """Parse data source dictionary."""
    # link yml keys to adapter classes
    ADAPTERS = {
        "RasterDataset": RasterDatasetAdapter,
        "GeoDataFrame": GeoDataFrameAdapter,
        "GeoDataset": GeoDatasetAdapter,
        "DataFrame": DataFrameAdapter,
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
        # Only for local files
        path = source.pop("path")
        # if remote path, keep as is else call abs_path method to solve local files
        if not _uri_validator(path):
            path = abs_path(root, path)
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
            # pop avoid placeholders being passed to adapter
            options = source.pop("placeholders")
            for combination in itertools.product(*options.values()):
                path_n = path
                name_n = name
                for k, v in zip(options.keys(), combination):
                    path_n = path_n.replace("{" + k + "}", v)
                    name_n = name_n.replace("{" + k + "}", v)
                data[name_n] = adapter(
                    path=path_n,
                    name=name_n,
                    catalog_name=catalog_name,
                    meta=meta,
                    **source,
                )
        else:
            data[name] = adapter(
                path=path, name=name, catalog_name=catalog_name, meta=meta, **source
            )

    return data


def _yml_from_uri_or_path(uri_or_path: Union[Path, str]) -> Dict:
    if _uri_validator(uri_or_path):
        with requests.get(uri_or_path, stream=True) as r:
            if r.status_code != 200:
                raise IOError(f"URL {r.content}: {uri_or_path}")
            yml = yaml.load(r.text, Loader=yaml.FullLoader)
    else:
        with open(uri_or_path, "r") as stream:
            yml = yaml.load(stream, Loader=yaml.FullLoader)
    return yml


def _process_dict(d: Dict, logger=logger) -> Dict:
    """Recursively change dict values to keep only python literal structures."""
    for k, v in d.items():
        _check_key = isinstance(k, str)
        if _check_key and isinstance(v, dict):
            d[k] = _process_dict(v)
        elif _check_key and isinstance(v, Path):
            d[k] = str(v)  # path to string
        # elif not _check_key or not isinstance(v, (list, str, int, float, bool)):
        #     d.pop(k)  # remove this entry
        #     logger.debug(f'Removing non-serializable entry "{k}"')
    return d


def abs_path(root: Union[Path, str], rel_path: Union[Path, str]) -> str:
    path = Path(str(rel_path))
    if not path.is_absolute():
        if root is not None:
            rel_path = join(root, rel_path)
        path = Path(abspath(rel_path))
    return str(path)
