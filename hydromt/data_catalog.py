#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DataCatalog module for HydroMT."""
from __future__ import annotations

import copy
import itertools
import logging
import os
import shutil
import warnings
from os.path import abspath, basename, exists, isdir, isfile, join
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
import yaml
from packaging.version import Version

from hydromt.utils import partition_dictionaries

from . import __version__
from .data_adapter import (
    DataAdapter,
    DataFrameAdapter,
    GeoDataFrameAdapter,
    GeoDatasetAdapter,
    RasterDatasetAdapter,
)
from .data_adapter.caching import HYDROMT_DATADIR, _copyfile, _uri_validator

logger = logging.getLogger(__name__)

__all__ = [
    "DataCatalog",
]

# just for typehints
SourceSpecDict = TypedDict(
    "SourceSpecDict", {"source": str, "provider": str, "version": Union[str, int]}
)


class DataCatalog(object):

    """Base class for the data catalog object."""

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
        """Catalog of DataAdapter sources.

        Helps to easily read from different files and keep track of
        files which have been accessed.

        Arguments
        ---------
        data_libs: (list of) str, Path, optional
            One or more paths to data catalog yaml files or names of predefined data
            catalogs. By default the data catalog is initiated without data entries.
            See :py:func:`~hydromt.data_adapter.DataCatalog.from_yml` for
            accepted yaml format.
        fallback_lib:
            Name of pre-defined data catalog to read if no data_libs are provided,
            by default 'artifact_data'.
            If None, no default data catalog is used.
        cache: bool, optional
            Set to true to cache data locally before reading.
            Currently only implemented for tiled rasterdatasets, by default False.
        cache_dir: str, Path, optional
            Folder root path to cach data to, by default ~/.hydromt_data
        artifact_keys:
            Deprecated from version v0.5
        logger : logger object, optional
            The logger object used for logging messages. If not provided, the default
            logger will be used.
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
    def keys(self) -> List[str]:
        """Returns list of data source names."""
        warnings.warn(
            "Using iterating over the DataCatalog directly is deprecated."
            "Please use cat.get_source()",
            DeprecationWarning,
        )
        return list(self._sources.keys())

    def get_source_names(self) -> List[str]:
        """Return a list of all available data source names."""
        return list(self._sources.keys())

    @property
    def predefined_catalogs(self) -> Dict:
        """Return all predefined catalogs."""
        if not self._catalogs:
            self.set_predefined_catalogs()
        return self._catalogs

    def get_source(
        self, source: str, provider: Optional[str] = None, version: Optional[str] = None
    ) -> DataAdapter:
        """Return a data source.

        Parameters
        ----------
        source : str
            Name of the data source.
        provider : str, optional
            Name of the data provider, by default None.
            By default the last added provider is returned.
        version : str, optional
            Version of the data source, by default None.
            By default the newest version of the requested provider is returned.

        Returns
        -------
        DataAdapter
            DataAdapter object.
        """
        source = str(source)
        if source not in self._sources:
            available_sources = sorted(list(self._sources.keys()))
            raise KeyError(
                f"Requested unknown data source '{source}' "
                f"available sources are: {available_sources}"
            )
        available_providers = self._sources[source]

        # make sure all arguments are strings
        provider = str(provider) if provider is not None else provider
        version = str(version) if version is not None else version

        # find provider matching requested version
        if provider is None and version is not None:
            providers = [p for p, v in available_providers.items() if version in v]
            if len(providers) > 0:  # error raised later if no provider found
                provider = providers[-1]

        # check if provider is available, otherwise use last added provider
        if provider is None:
            requested_provider = list(available_providers.keys())[-1]
        else:
            requested_provider = provider
            if requested_provider not in available_providers:
                providers = sorted(list(available_providers.keys()))
                raise KeyError(
                    f"Requested unknown provider '{requested_provider}' for "
                    f"data source '{source}' available providers are: {providers}"
                )
        available_versions = available_providers[requested_provider]

        # check if version is available, otherwise use last added version which is
        # always the newest version
        if version is None:
            requested_version = list(available_versions.keys())[-1]
        else:
            requested_version = version
            if requested_version not in available_versions:
                data_versions = sorted(list(map(str, available_versions.keys())))
                raise KeyError(
                    f"Requested unknown version '{requested_version}' for "
                    f"data source '{source}' and provider '{requested_provider}' "
                    f"available versions are {data_versions}"
                )

        return self._sources[source][requested_provider][requested_version]

    def add_source(self, source: str, adapter: DataAdapter) -> None:
        """Add a new data source to the data catalog.

        The data version and provider are extracted from the DataAdapter object.

        Parameters
        ----------
        source : str
            Name of the data source.
        adapter : DataAdapter
            DataAdapter object.
        """
        if not isinstance(adapter, DataAdapter):
            raise ValueError("Value must be DataAdapter")

        if hasattr(adapter, "version") and adapter.version is not None:
            version = adapter.version
        else:
            version = "_UNSPECIFIED_"  # make sure this comes first in sorted list

        if hasattr(adapter, "provider") and adapter.provider is not None:
            provider = adapter.provider
        else:
            provider = adapter.catalog_name

        if source not in self._sources:
            self._sources[source] = {}
        else:  # check if data type is the same as adapter with same name
            adapter0 = next(iter(next(iter(self._sources[source].values())).values()))
            if adapter0.data_type != adapter.data_type:
                raise ValueError(
                    f"Data source '{source}' already exists with data type "
                    f"'{adapter0.data_type}' but new data source has data type "
                    f"'{adapter.data_type}'."
                )

        if provider not in self._sources[source]:
            versions = {version: adapter}
        else:
            versions = self._sources[source][provider]
            if provider in self._sources[source] and version in versions:
                warnings.warn(
                    f"overwriting data source '{source}' with "
                    f"provider {provider} and version {version}.",
                    UserWarning,
                )
            # update and sort dictionary -> make sure newest version is last
            versions.update({version: adapter})
            versions = {k: versions[k] for k in sorted(list(versions.keys()))}

        self._sources[source][provider] = versions

    def __getitem__(self, key: str) -> DataAdapter:
        """Get the source."""
        warnings.warn(
            'Using iterating over the DataCatalog directly is deprecated."\
            " Please use cat.get_source("name")',
            DeprecationWarning,
        )
        return self.get_source(key)

    def __setitem__(self, key: str, value: DataAdapter) -> None:
        """Set or update adaptors."""
        warnings.warn(
            "Using DataCatalog as a dictionary directly is deprecated."
            " Please use cat.add_source(adapter)",
            DeprecationWarning,
        )
        self.add_source(key, value)

    def iter_sources(self) -> List[Tuple[str, DataAdapter]]:
        """Return a flat list of all available data sources with no duplicates."""
        ans = []
        for source_name, available_providers in self._sources.items():
            for _, available_versions in available_providers.items():
                for _, adapter in available_versions.items():
                    ans.append((source_name, adapter))

        return ans

    def __iter__(self):
        """Iterate over sources."""
        warnings.warn(
            "Using iterating over the DataCatalog directly is deprecated."
            " Please use cat.iter_sources()",
            DeprecationWarning,
        )
        return self.iter_sources()

    def __len__(self):
        """Return number of sources."""
        return len(self.iter_sources())

    def __repr__(self):
        """Prettyprint the sources."""
        return self.to_dataframe().__repr__()

    def __eq__(self, other) -> bool:
        if type(other) is type(self):
            if len(self) != len(other):
                return False
            for name, source in self.iter_sources():
                try:
                    other_source = other.get_source(
                        name, provider=source.provider, version=source.version
                    )
                except KeyError:
                    return False
                if source != other_source:
                    return False
        else:
            return False
        return True

    def _repr_html_(self):
        return self.to_dataframe()._repr_html_()

    def update(self, **kwargs) -> None:
        """Add data sources to library or update them."""
        for k, v in kwargs.items():
            self.add_source(k, v)

    def update_sources(self, **kwargs) -> None:
        """Add data sources to library or update them."""
        self.update(**kwargs)

    def set_predefined_catalogs(self, urlpath: Union[Path, str] = None) -> Dict:
        """Initialise the predefined catalogs."""
        # get predefined_catalogs
        urlpath = self._url if urlpath is None else urlpath
        cache_path = join(self._cache_dir, basename(urlpath))
        try:
            # download file locally; overwrite existing file
            _copyfile(urlpath, cache_path)
        except Exception:  # if offline
            self.logger.warning(
                "Downloading the predefined catalogs failed;"
                "check your internet connection"
            )
            pass
        if isfile(cache_path):
            self._catalogs = _yml_from_uri_or_path(cache_path)
        if self._catalogs is None:
            raise ConnectionError(
                "Predefined catalogs not found; check your internet connection."
            )
        return self._catalogs

    def from_artifacts(
        self, name: str = "artifact_data", version: str = "latest"
    ) -> DataCatalog:
        """Parse artifacts.

        Deprecated method. Use
        :py:func:`hydromt.data_catalog.DataCatalog.from_predefined_catalogs` instead.

        Parameters
        ----------
        name : str, optional
            Catalog name. If None (default) sample data is downloaded.
        version : str, optional
            Release version. By default it takes the latest known release.

        Returns
        -------
        DataCatalog
            DataCatalog object with parsed artifact data.
        """
        warnings.warn(
            '"from_artifacts" is deprecated. Use "from_predefined_catalogs instead".',
            DeprecationWarning,
        )
        return self.from_predefined_catalogs(name, version)

    def from_predefined_catalogs(
        self, name: str, version: str = "latest"
    ) -> DataCatalog:
        """Add data sources from a predefined data catalog.

        Parameters
        ----------
        name : str
            Catalog name.
        version : str, optional
            Catlog release version. By default it takes the latest known release.

        Returns
        -------
        DataCatalog
            DataCatalog object with parsed predefined catalog added.
        """
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
            self.from_yml(urlpath, catalog_name=name)

    def from_archive(
        self, urlpath: Union[Path, str], version: str = None, name: str = None
    ) -> DataCatalog:
        """Read a data archive including a data_catalog.yml file.

        Parameters
        ----------
        urlpath : str, Path
            Path or url to data archive.
        version : str, optional
            Version of data archive, by default None.
        name : str, optional
            Name of data catalog, by default None.

        Returns
        -------
        DataCatalog
            DataCatalog object with parsed data archive added.
        """
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
        return self.from_yml(yml_fn, catalog_name=name)

    def from_yml(
        self,
        urlpath: Union[Path, str],
        root: str = None,
        catalog_name: str = None,
        mark_used: bool = False,
    ) -> DataCatalog:
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
              driver_kwargs:
                <key>: <value>
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

        Returns
        -------
        DataCatalog
            DataCatalog object with parsed yaml file added.
        """
        self.logger.info(f"Parsing data catalog from {urlpath}")
        yml = _yml_from_uri_or_path(urlpath)
        # parse metadata
        meta = dict()
        # legacy code with root/category at highest yml level
        if "root" in yml:
            warnings.warn(
                "The 'root' key is deprecated, use 'meta: root' instead.",
                DeprecationWarning,
            )
            meta.update(root=yml.pop("root"))
        if "category" in yml:
            warnings.warn(
                "The 'category' key is deprecated, use 'meta: category' instead.",
                DeprecationWarning,
            )
            meta.update(category=yml.pop("category"))

        # read meta data
        meta = yml.pop("meta", meta)
        # check version required hydromt version
        hydromt_version = meta.get("hydromt_version", __version__)
        self_version = Version(__version__)
        yml_version = Version(hydromt_version)
        if yml_version > self_version:
            self.logger.warning(
                f"Specified HydroMT version ({hydromt_version}) \
                  more recent than installed version ({__version__}).",
            )
        if catalog_name is None:
            catalog_name = meta.get("name", "".join(basename(urlpath).split(".")[:-1]))
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
    ) -> DataCatalog:
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
        A data dictionary with two entries is provided below, where all the text between
        <> should be filled by the user. See the specific data adapters
        for more information about the required and optional arguments.

        .. code-block:: text

            {
                <name1>: {
                    "path": <path>,
                    "data_type": <data_type>,
                    "driver": <driver>,
                    "filesystem": <filesystem>,
                    "driver_kwargs": {<key>: <value>},
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
        meta = data_dict.pop("meta", {})
        if "root" in meta and root is None:
            root = meta.pop("root")
        if "category" in meta and category is None:
            category = meta.pop("category")
        if "name" in meta and catalog_name is None:
            catalog_name = meta.pop("name")
        for name, source_dict in _denormalise_data_dict(data_dict):
            adapter = _parse_data_source_dict(
                name,
                source_dict,
                catalog_name=catalog_name,
                root=root,
                category=category,
            )
            self.add_source(name, adapter)
            if mark_used:
                self._used_data.append(name)

        return self

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
            If "auto" (default) the data source paths are relative to the yaml
            output ``path``.
        source_names: list, optional
            List of source names to export, by default None in which case all sources
            are exported. This argument is ignored if `used_only=True`.
        used_only: bool, optional
            If True, export only data entries kept in used_data list, by default False.
        meta: dict, optional
            key-value pairs to add to the data catalog meta section, such as 'version',
            by default empty.
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
            List of source names to export, by default None in which case all sources
            are exported.
        root : str, Path, optional
            Global root for all relative paths in yml file.
        meta: dict, optional
            key-value pairs to add to the data catalog meta section, such as 'version',
            by default empty.

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
        sorted_sources = sorted(self.iter_sources(), key=lambda x: x[0])
        for name, source in sorted_sources:  # alphabetical order
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
            if name in sources_out:
                existing = sources_out.pop(name)
                if existing == source_dict:
                    sources_out.update({name: source_dict})
                    continue
                if "variants" in existing:
                    variants = existing.pop("variants")
                    _, variant, _ = partition_dictionaries(source_dict, existing)
                    variants.append(variant)
                    existing["variants"] = variants
                else:
                    base, diff_existing, diff_new = partition_dictionaries(
                        source_dict, existing
                    )
                    # provider and version should always be in variants list
                    provider = base.pop("provider", None)
                    if provider is not None:
                        diff_existing["provider"] = provider
                        diff_new["provider"] = provider
                    version = base.pop("version", None)
                    if version is not None:
                        diff_existing["version"] = version
                        diff_new["version"] = version
                    base["variants"] = [diff_new, diff_existing]
                sources_out[name] = base
            else:
                sources_out.update({name: source_dict})
        if meta:
            sources_out = {"meta": meta, **sources_out}
        return sources_out

    def to_dataframe(self, source_names: List = []) -> pd.DataFrame:
        """Return data catalog summary as DataFrame."""
        d = []
        for name, source in self.iter_sources():
            if len(source_names) > 0 and name not in source_names:
                continue
            d.append(
                {
                    "name": name,
                    "provider": source.provider,
                    "version": source.version,
                    **source.summary(),
                }
            )
        return pd.DataFrame.from_records(d).set_index("name")

    def export_data(
        self,
        data_root: Union[Path, str],
        bbox: List = None,
        time_tuple: Tuple = None,
        source_names: List = [],
        unit_conversion: bool = True,
        meta: Dict = {},
        append: bool = False,
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
            List of source names to export, by default None in which case all sources
            are exported. Specific variables can be selected by appending them to the
            source name in square brackets. For example, to export all variables of
            'source_name1' and only 'var1' and 'var2' of 'source_name'
            use source_names=['source_name1', 'source_name2[var1,var2]']
        unit_conversion: boolean, optional
            If False skip unit conversion when parsing data from file, by default True.
        meta: dict, optional
            key-value pairs to add to the data catalog meta section, such as 'version',
            by default empty.
        append: bool, optional
            If True, append to existing data catalog, by default False.
        """
        data_root = abspath(data_root)
        if not os.path.isdir(data_root):
            os.makedirs(data_root)

        # create copy of data with selected source names
        source_vars = {}
        if len(source_names) > 0:
            sources = {}
            for name in source_names:
                # deduce variables from name
                if "[" in name:
                    variables = name.split("[")[-1].split("]")[0].split(",")
                    name = name.split("[")[0]
                    source_vars[name] = variables

                source = self.get_source(name)
                provider = source.provider
                version = source.version

                if name not in sources:
                    sources[name] = {}
                if provider not in sources[name]:
                    sources[name][provider] = {}

                sources[name][provider][version] = copy.deepcopy(source)

        else:
            sources = copy.deepcopy(self.sources)

        # read existing data catalog if it exists
        fn = join(data_root, "data_catalog.yml")
        if isfile(fn) and append:
            self.logger.info(f"Appending existing data catalog {fn}")
            sources_out = DataCatalog(fn).sources
        else:
            sources_out = {}

        # export data and update sources
        for key, available_variants in sources.items():
            for provider, available_versions in available_variants.items():
                for version, source in available_versions.items():
                    try:
                        # read slice of source and write to file
                        self.logger.debug(f"Exporting {key}.")
                        if not unit_conversion:
                            unit_mult = source.unit_mult
                            unit_add = source.unit_add
                            source.unit_mult = {}
                            source.unit_add = {}
                        fn_out, driver, source_kwargs = source.to_file(
                            data_root=data_root,
                            data_name=key,
                            variables=source_vars.get(key, None),
                            bbox=bbox,
                            time_tuple=time_tuple,
                            logger=self.logger,
                        )
                        if fn_out is None:
                            self.logger.warning(
                                f"{key} file contains no data within domain"
                            )
                            continue
                        # update path & driver and remove kwargs
                        # and rename in output sources
                        if unit_conversion:
                            source.unit_mult = {}
                            source.unit_add = {}
                        else:
                            source.unit_mult = unit_mult
                            source.unit_add = unit_add
                        source.path = fn_out
                        source.driver = driver
                        source.filesystem = "local"
                        source.driver_kwargs = {}
                        source.rename = {}
                        if key in sources_out:
                            self.logger.warning(
                                f"{key} already exists in data catalog, overwriting..."
                            )
                        if key not in sources_out:
                            sources_out[key] = {}
                        if provider not in sources_out[key]:
                            sources_out[key][provider] = {}

                        sources_out[key][provider][version] = source
                    except FileNotFoundError:
                        self.logger.warning(f"{key} file not found at {source.path}")

        # write data catalog to yml
        data_catalog_out = DataCatalog()
        for key, available_variants in sources_out.items():
            for provider, available_versions in available_variants.items():
                for version, adapter in available_versions.items():
                    data_catalog_out.add_source(key, adapter)

        data_catalog_out.to_yml(fn, root="auto", meta=meta)

    def get_rasterdataset(
        self,
        data_like: Union[str, SourceSpecDict, Path, xr.Dataset, xr.DataArray],
        bbox: Optional[List] = None,
        geom: Optional[gpd.GeoDataFrame] = None,
        zoom_level: Optional[int | tuple] = None,
        buffer: Union[float, int] = 0,
        align: Optional[bool] = None,
        variables: Optional[Union[List, str]] = None,
        time_tuple: Optional[Tuple] = None,
        single_var_as_array: Optional[bool] = True,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Return a clipped, sliced and unified RasterDataset.

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` and `align` arguments.
        To slice the data to the time period of interest, provide the `time_tuple`
        argument. To return only the dataset variables of interest provide the
        `variables` argument.

        NOTE: Unless `single_var_as_array` is set to False a single-variable data source
        will be returned as :py:class:`xarray.DataArray` rather than
        :py:class:`xarray.Dataset`.

        Arguments
        ---------
        data_like: str, Path, Dict, xr.Dataset, xr.Datarray
            DataCatalog key, path to raster file or raster xarray data object.
            The catalog key can be a string or a dictionary with the following keys:
            {'name', 'provider', 'version'}.
            If a path to a raster file is provided it will be added
            to the catalog with its based on the file basename.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest
            (in WGS84 coordinates).
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        zoom_level : int, tuple, optional
            Zoom level of the xyz tile dataset (0 is base level)
            Using a tuple the zoom level can be specified as
            (<zoom_resolution>, <unit>), e.g., (1000, 'meter')
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
        provider: str, optional
            Data source provider. If None (default) the last added provider is used.
        version: str, optional
            Data source version. If None (default) the newest version is used.
        **kwargs:
            Additional keyword arguments that are passed to the `RasterDatasetAdapter`
            function. Only used if `data_like` is a path to a raster file.

        Returns
        -------
        obj: xarray.Dataset or xarray.DataArray
            RasterDataset
        """
        if isinstance(data_like, dict):
            data_like, provider, version = _parse_data_like_dict(
                data_like, provider, version
            )

        if isinstance(data_like, (str, Path)):
            if isinstance(data_like, str) and data_like in self.sources:
                name = data_like
                source = self.get_source(name, provider=provider, version=version)
            elif exists(abspath(data_like)):
                path = str(abspath(data_like))
                if "provider" not in kwargs:
                    kwargs.update({"provider": "local"})
                source = RasterDatasetAdapter(path=path, **kwargs)
                name = basename(data_like)
                self.add_source(name, source)
            else:
                raise FileNotFoundError(f"No such file or catalog source: {data_like}")
        elif isinstance(data_like, (xr.DataArray, xr.Dataset)):
            # TODO apply bbox, geom, buffer, align, variables, time_tuple
            return data_like
        else:
            raise ValueError(f'Unknown raster data type "{type(data_like).__name__}"')

        self._used_data.append(name)
        self.logger.info(
            f"DataCatalog: Getting {name} RasterDataset {source.driver} data from"
            f" {source.path}"
        )
        obj = source.get_data(
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
        data_like: Union[str, SourceSpecDict, Path, xr.Dataset, xr.DataArray],
        bbox: Optional[List] = None,
        geom: Optional[gpd.GeoDataFrame] = None,
        buffer: Union[float, int] = 0,
        variables: Optional[Union[List, str]] = None,
        predicate: str = "intersects",
        provider: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ):
        """Return a clipped and unified GeoDataFrame (vector).

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` and `align` arguments.
        To return only the dataframe columns of interest provide the
        `variables` argument.

        Arguments
        ---------
        data_like: str, Path, gpd.GeoDataFrame
            Data catalog key, path to vector file or a vector geopandas object.
            The catalog key can be a string or a dictionary with the following keys:
            {'name', 'provider', 'version'}.
            If a path to a vector file is provided it will be added
            to the catalog with its based on the file basename.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest
            (in WGS84 coordinates).
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        buffer : float, optional
            Buffer around the `bbox` or `geom` area of interest in meters. By default 0.
        predicate : {'intersects', 'within', 'contains', 'overlaps',
            'crosses', 'touches'}, optional If predicate is provided,
            the GeoDataFrame is filtered by testing the predicate function
            against each item. Requires bbox or mask. By default 'intersects'
        align : float, optional
            Resolution to align the bounding box, by default None
        variables : str or list of str, optional.
            Names of GeoDataFrame columns to return. By default all columns are
            returned.
        provider: str, optional
            Data source provider. If None (default) the last added provider is used.
        version: str, optional
            Data source version. If None (default) the newest version is used.
        **kwargs:
            Additional keyword arguments that are passed to the `GeoDataFrameAdapter`
            function. Only used if `data_like` is a path to a vector file.

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame
        """
        if isinstance(data_like, dict):
            data_like, provider, version = _parse_data_like_dict(
                data_like, provider, version
            )
        if isinstance(data_like, (str, Path)):
            if str(data_like) in self.sources:
                name = data_like
                source = self.get_source(name, provider=provider, version=version)
            elif exists(abspath(data_like)):
                path = str(abspath(data_like))
                if "provider" not in kwargs:
                    kwargs.update({"provider": "local"})
                source = GeoDataFrameAdapter(path=path, **kwargs)
                name = basename(data_like)
                self.add_source(name, source)
            else:
                raise FileNotFoundError(f"No such file or catalog source: {data_like}")
        elif isinstance(data_like, gpd.GeoDataFrame):
            # TODO apply bbox, geom, buffer, predicate, variables
            return data_like
        else:
            raise ValueError(f'Unknown vector data type "{type(data_like).__name__}"')

        self._used_data.append(name)
        self.logger.info(
            f"DataCatalog: Getting {name} GeoDataFrame {source.driver} data"
            f" from {source.path}"
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
        data_like: Union[str, SourceSpecDict, Path, xr.Dataset, xr.DataArray],
        bbox: Optional[List] = None,
        geom: Optional[gpd.GeoDataFrame] = None,
        buffer: Union[float, int] = 0,
        variables: Optional[List] = None,
        time_tuple: Optional[Tuple] = None,
        single_var_as_array: bool = True,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Return a clipped, sliced and unified GeoDataset.

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` and `align` arguments.
        To slice the data to the time period of interest, provide the
        `time_tuple` argument. To return only the dataset variables
        of interest provide the `variables` argument.

        NOTE: Unless `single_var_as_array` is set to False a single-variable data source
        will be returned as xarray.DataArray rather than Dataset.

        Arguments
        ---------
        data_like: str, Path, xr.Dataset, xr.DataArray
            Data catalog key, path to geodataset file or geodataset xarray object.
            The catalog key can be a string or a dictionary with the following keys:
            {'name', 'provider', 'version'}.
            If a path to a file is provided it will be added
            to the catalog with its based on the file basename.
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box of area of interest
            (in WGS84 coordinates).
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        buffer : float, optional
            Buffer around the `bbox` or `geom` area of interest in meters. By default 0.
        variables : str or list of str, optional.
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        single_var_as_array: bool, optional
            If True, return a DataArray if the dataset consists of a single variable.
            If False, always return a Dataset. By default True.
        **kwargs:
            Additional keyword arguments that are passed to the `GeoDatasetAdapter`
            function. Only used if `data_like` is a path to a geodataset file.

        Returns
        -------
        obj: xarray.Dataset or xarray.DataArray
            GeoDataset
        """
        if isinstance(data_like, dict):
            data_like, provider, version = _parse_data_like_dict(
                data_like, provider, version
            )
        if isinstance(data_like, (str, Path)):
            if isinstance(data_like, str) and data_like in self.sources:
                name = data_like
                source = self.get_source(name, provider=provider, version=version)
            elif exists(abspath(data_like)):
                path = str(abspath(data_like))
                if "provider" not in kwargs:
                    kwargs.update({"provider": "local"})
                source = GeoDatasetAdapter(path=path, **kwargs)
                name = basename(data_like)
                self.add_source(name, source)
            else:
                raise FileNotFoundError(f"No such file or catalog source: {data_like}")
        elif isinstance(data_like, (xr.DataArray, xr.Dataset)):
            # TODO apply bbox, geom, buffer, variables, time_tuple
            return data_like
        else:
            raise ValueError(f'Unknown geo data type "{type(data_like).__name__}"')

        self._used_data.append(name)
        self.logger.info(
            f"DataCatalog: Getting {name} GeoDataset {source.driver} data"
            f" from {source.path}"
        )
        obj = source.get_data(
            bbox=bbox,
            geom=geom,
            buffer=buffer,
            variables=variables,
            time_tuple=time_tuple,
            single_var_as_array=single_var_as_array,
        )
        return obj

    def get_dataframe(
        self,
        data_like: Union[str, SourceSpecDict, Path, xr.Dataset, xr.DataArray],
        variables: Optional[list] = None,
        time_tuple: Optional[Tuple] = None,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ):
        """Return a unified and sliced DataFrame.

        Parameters
        ----------
        data_like : str, Path, pd.DataFrame
            Data catalog key, path to tabular data file or tabular pandas dataframe.
            The catalog key can be a string or a dictionary with the following keys:
            {'name', 'provider', 'version'}.
            If a path to a tabular data file is provided it will be added
            to the catalog with its based on the file basename.
        variables : str or list of str, optional.
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        **kwargs:
            Additional keyword arguments that are passed to the `DataframeAdapter`
            function. Only used if `data_like` is a path to a tabular data file.

        Returns
        -------
        pd.DataFrame
            Tabular data
        """
        if isinstance(data_like, dict):
            data_like, provider, version = _parse_data_like_dict(
                data_like, provider, version
            )
        if isinstance(data_like, (str, Path)):
            if isinstance(data_like, str) and data_like in self.sources:
                name = data_like
                source = self.get_source(name, provider=provider, version=version)
            elif exists(abspath(data_like)):
                path = str(abspath(data_like))
                if "provider" not in kwargs:
                    kwargs.update({"provider": "local"})
                source = DataFrameAdapter(path=path, **kwargs)
                name = basename(data_like)
                self.add_source(name, source)
            else:
                raise FileNotFoundError(f"No such file or catalog source: {data_like}")
        elif isinstance(data_like, pd.DataFrame):
            return data_like
        else:
            raise ValueError(f'Unknown tabular data type "{type(data_like).__name__}"')

        self._used_data.append(name)
        self.logger.info(
            f"DataCatalog: Getting {name} DataFrame {source.driver} data"
            f" from {source.path}"
        )
        obj = source.get_data(
            variables=variables,
            time_tuple=time_tuple,
            logger=self.logger,
        )
        return obj


def _parse_data_like_dict(
    data_like: SourceSpecDict,
    provider: Optional[str] = None,
    version: Optional[str] = None,
):
    if not SourceSpecDict.__required_keys__.issuperset(set(data_like.keys())):
        unknown_keys = set(data_like.keys()) - SourceSpecDict.__required_keys__
        raise ValueError(f"Unknown keys in requested data source: {unknown_keys}")
    elif "source" not in data_like:
        raise ValueError("No source key found in requested data source")
    else:
        source = data_like.get("source")
        provider = data_like.get("provider", provider)
        version = data_like.get("version", version)
    return source, provider, version


def _parse_data_source_dict(
    name: str,
    data_source_dict: Dict,
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
    # parse data
    source = data_source_dict.copy()  # important as we modify with pop

    # parse path
    if "path" not in source:
        raise ValueError(f"{name}: Missing required path argument.")
    # if remote path, keep as is else call abs_path method to solve local files
    path = source.pop("path")
    if not _uri_validator(str(path)):
        path = abs_path(root, path)
    # parse data type > adapter
    data_type = source.pop("data_type", None)
    if data_type is None:
        raise ValueError(f"{name}: Data type missing.")
    elif data_type not in ADAPTERS:
        raise ValueError(f"{name}: Data type {data_type} unknown")
    adapter = ADAPTERS.get(data_type)
    # source meta data
    meta = source.pop("meta", {})
    if "category" not in meta and category is not None:
        meta.update(category=category)

    # driver arguments
    driver_kwargs = source.pop("driver_kwargs", source.pop("kwargs", {}))
    for driver_kwarg in driver_kwargs:
        # required for geodataset where driver_kwargs can be a path
        if "fn" in driver_kwarg:
            driver_kwargs.update(
                {driver_kwarg: abs_path(root, driver_kwargs[driver_kwarg])}
            )

    return adapter(
        path=path,
        name=name,
        catalog_name=catalog_name,
        meta=meta,
        driver_kwargs=driver_kwargs,
        **source,
    )


def _yml_from_uri_or_path(uri_or_path: Union[Path, str]) -> Dict:
    if _uri_validator(str(uri_or_path)):
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
    return d


def _denormalise_data_dict(data_dict) -> List[Tuple[str, Dict]]:
    """Return a flat list of with data name, dictionary of input data_dict.

    Expand possible versions, aliases and variants in data_dict.
    """
    data_list = []
    for name, source in data_dict.items():
        source = copy.deepcopy(source)
        data_dicts = []
        if "alias" in source:
            alias = source.pop("alias")
            warnings.warn(
                "The use of alias is deprecated, please add a version on the aliased"
                "catalog instead.",
                DeprecationWarning,
            )
            if alias not in data_dict:
                raise ValueError(f"alias {alias} not found in data_dict.")
            # use alias source but overwrite any attributes with original source
            source_copy = data_dict[alias].copy()
            source_copy.update(source)
            data_dicts.append({name: source_copy})
        elif "variants" in source:
            variants = source.pop("variants")
            for diff in variants:
                source_copy = copy.deepcopy(source)
                source_copy.update(**diff)
                data_dicts.append({name: source_copy})
        elif "placeholders" in source:
            options = source.pop("placeholders")
            for combination in itertools.product(*options.values()):
                source_copy = copy.deepcopy(source)
                name_copy = name
                for k, v in zip(options.keys(), combination):
                    name_copy = name_copy.replace("{" + k + "}", v)
                    source_copy["path"] = source_copy["path"].replace("{" + k + "}", v)
                data_dicts.append({name_copy: source_copy})
        else:
            data_list.append((name, source))
            continue

        # recursively denormalise in case of multiple denormalise keys in source
        for item in data_dicts:
            data_list.extend(_denormalise_data_dict(item))

    return data_list


def abs_path(root: Union[Path, str], rel_path: Union[Path, str]) -> str:
    path = Path(str(rel_path))
    if not path.is_absolute():
        if root is not None:
            rel_path = join(root, rel_path)
        path = Path(abspath(rel_path))
    return str(path)
