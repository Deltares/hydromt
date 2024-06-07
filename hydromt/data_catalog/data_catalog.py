"""DataCatalog module for HydroMT."""

from __future__ import annotations

import copy
import itertools
import logging
import os
import warnings
from datetime import datetime
from os.path import abspath, basename, exists, isfile, join, splitext
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import geopandas as gpd
import numpy as np
import pandas as pd
import pooch
import xarray as xr
import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pystac import Catalog as StacCatalog
from pystac import CatalogType, MediaType

from hydromt import __version__
from hydromt._typing import Bbox, ErrorHandleMethod, SourceSpecDict, TimeRange
from hydromt._typing.error import NoDataException, NoDataStrategy, exec_nodata_strat
from hydromt._typing.type_def import StrPath
from hydromt._utils import _partition_dictionaries, _single_var_as_array
from hydromt.config import SETTINGS
from hydromt.data_catalog.adapters import (
    DataAdapter,
    DataFrameAdapter,
    DatasetAdapter,
    GeoDataFrameAdapter,
    GeoDatasetAdapter,
    RasterDatasetAdapter,
)
from hydromt.data_catalog.predefined_catalog import (
    PredefinedCatalog,
    _copy_file,
)
from hydromt.data_catalog.sources import (
    DataFrameSource,
    DataSource,
    GeoDataFrameSource,
    GeoDatasetSource,
    RasterDatasetSource,
    create_source,
)
from hydromt.gis.gis_utils import parse_geom_bbox_buffer
from hydromt.io.readers import _yml_from_uri_or_path
from hydromt.plugins import PLUGINS
from hydromt.utils import deep_merge

logger = logging.getLogger(__name__)

__all__ = ["DataCatalog"]


class DataCatalog(object):
    """Base class for the data catalog object."""

    _format_version = "v1"  # format version of the data catalog
    _cache_dir = SETTINGS.cache_root

    def __init__(
        self,
        data_libs: Optional[Union[List, str]] = None,
        fallback_lib: Optional[str] = "artifact_data",
        logger=logger,
        cache: Optional[bool] = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Catalog of DataAdapter sources.

        Helps to easily read from different files and keep track of
        files which have been accessed.

        Arguments
        ---------
        data_libs: (list of) str, Path, optional
            One or more paths to data catalog configuration files or names of predefined
            data catalogs. By default the data catalog is initiated without data
            entries. See :py:func:`~hydromt.data_catalog.adapters.DataCatalog.from_yml` for
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
        logger : logger object, optional
            The logger object used for logging messages. If not provided, the default
            logger will be used.
        """
        if data_libs is None:
            data_libs = []
        elif not isinstance(data_libs, list):  # make sure data_libs is a list
            data_libs = np.atleast_1d(data_libs).tolist()

        data_libs = cast(list, data_libs)

        self._sources = {}  # dictionary of DataSource
        self._catalogs: Dict[str, PredefinedCatalog] = {}
        self.root = None
        self._fallback_lib = fallback_lib
        self.logger = logger

        # caching
        self.cache = bool(cache)
        if cache_dir is not None:
            self._cache_dir = cache_dir

        for name_or_path in data_libs:
            if str(name_or_path).split(".")[-1] in ["yml", "yaml"]:  # user defined
                self.from_yml(name_or_path)
            else:  # predefined
                self.from_predefined_catalogs(name_or_path)

    @property
    def sources(self) -> Dict[DataSource]:
        """Returns dictionary of DataSources."""
        if len(self._sources) == 0 and self._fallback_lib is not None:
            # read artifacts by default if no catalogs are provided
            self.from_predefined_catalogs(self._fallback_lib)
        return self._sources

    def get_source_names(self) -> List[str]:
        """Return a list of all available data source names."""
        return list(self._sources.keys())

    def to_stac_catalog(
        self,
        root: Union[str, Path],
        source_names: Optional[List] = None,
        meta: Optional[Dict] = None,
        catalog_name: str = "hydromt-stac-catalog",
        description: str = "The stac catalog of hydromt",
        used_only: bool = False,
        catalog_type: CatalogType = CatalogType.RELATIVE_PUBLISHED,
        on_error: ErrorHandleMethod = ErrorHandleMethod.COERCE,
    ):
        """Write data catalog to STAC format.

        Parameters
        ----------
        path: str, Path
            stac output path.
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
        meta = meta or {}
        stac_catalog = StacCatalog(id=catalog_name, description=description)
        for _name, source in self.list_sources(used_only):
            stac_child_catalog = source.to_stac_catalog(on_error)
            if stac_child_catalog:
                stac_catalog.add_child(stac_child_catalog)

        stac_catalog.normalize_and_save(root, catalog_type=catalog_type)
        return stac_catalog

    def from_stac_catalog(
        self,
        stac_like: Union[str, Path, StacCatalog, dict],
        on_error: ErrorHandleMethod = ErrorHandleMethod.SKIP,
    ):
        """Write data catalog to STAC format.

        Parameters
        ----------
        path: str, Path
            stac path.
        on_error: ErrorHandleMethod
            What to do on error when converting from STAC
        """
        if isinstance(stac_like, (str, Path)):
            stac_catalog = StacCatalog.from_file(stac_like)
        elif isinstance(stac_like, dict):
            stac_catalog = StacCatalog.from_dict(stac_like)
        elif isinstance(stac_like, StacCatalog):
            stac_catalog = stac_like
        else:
            raise ValueError(
                f"Unsupported type for stac_like: {type(stac_like).__name__}"
            )

        for item in stac_catalog.get_items(recursive=True):
            source_name = item.id
            for _asset_name, asset in item.get_assets().items():
                if asset.media_type in [
                    MediaType.HDF,
                    MediaType.HDF5,
                    MediaType.COG,
                    MediaType.TIFF,
                ]:
                    Source: Type[DataSource] = RasterDatasetSource
                elif asset.media_type in [MediaType.GEOPACKAGE, MediaType.FLATGEOBUF]:
                    Source: Type[DataSource] = GeoDataFrameSource
                elif asset.media_type == MediaType.GEOJSON:
                    Source: Type[DataSource] = GeoDatasetSource
                elif asset.media_type == MediaType.JSON:
                    Source: Type[DataSource] = DataFrameSource
                else:
                    continue

                source: DataSource = Source(
                    name=source_name,
                    uri=asset.get_absolute_href(),
                    driver=Source._fallback_driver_read,
                )
                self.add_source(source_name, source)

        return self

    @property
    def predefined_catalogs(self) -> Dict:
        """Return all predefined catalogs."""
        if not self._catalogs:
            self._set_predefined_catalogs()
        return self._catalogs

    def get_source_bbox(
        self,
        source: str,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        detect: bool = True,
        strict: bool = False,
    ) -> Optional[Tuple[Tuple[float, float, float, float], int]]:
        """Retrieve the bounding box and crs of the source.

        Parameters
        ----------
        source: str,
            the name of the data source.
        provider: Optional[str]
            the provider of the source to detect the bbox of, if None, the last one
            added will be used.
        version: Optional[str]
            the version of the source to detect the bbox of, if None, the last one
            added will be used.
        detect: bool
            Whether to detect the bbox of the source if it is not set.
        strict: bool
            Raise an error if the adapter does not support bbox detection (such as
            dataframes). In that case, a warning will be logged instead.

        Returns
        -------
        bbox: Tuple[np.float64,np.float64,np.float64,np.float64]
            the bounding box coordinates of the data. coordinates are returned as
            [xmin,ymin,xmax,ymax]
        crs: int
            The ESPG code of the CRS of the coordinates returned in bbox
        """
        s: DataSource = self.get_source(source, provider, version)
        try:
            return s.get_bbox(detect=detect)  # type: ignore
        except TypeError as e:
            if strict:
                raise e
            else:
                self.logger.warning(
                    f"Source of type {type(s)} does not support detecting spatial"
                    "extents. skipping..."
                )

    def get_source_time_range(
        self,
        source: str,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        detect: bool = True,
        strict: bool = False,
    ) -> Optional[Tuple[datetime, datetime]]:
        """Detect the temporal range of the dataset.

        Parameters
        ----------
        source: str,
            the name of the data source.
        provider: Optional[str]
            the provider of the source to detect the time range of, if None,
            the last one added will be used.
        version: Optional[str]
            the version of the source to detect the time range of, if None, the last one
            added will be used.
        detect: bool
            Whether to detect the time range of the source if it is not set.
        strict: bool
            Raise an error if the adapter does not support time range detection (such as
            dataframes). In that case, a warning will be logged instead.

        Returns
        -------
        range: Tuple[np.datetime64, np.datetime64]
            A tuple containing the start and end of the time dimension. Range is
            inclusive on both sides.
        """
        s = self.get_source(source, provider, version)
        try:
            return s.get_time_range(detect=detect)  # type: ignore
        except TypeError as e:
            if strict:
                raise e
            else:
                self.logger.warning(
                    f"Source of type {type(s)} does not support detecting"
                    " temporalextents. skipping..."
                )

    def get_source(
        self,
        source: str,
        provider: Optional[str] = None,
        version: Optional[str] = None,
    ) -> DataSource:
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
        DataSource
            DataSource object.
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
                versions = sorted(list(map(str, available_versions.keys())))
                raise KeyError(
                    f"Requested unknown version '{requested_version}' for "
                    f"data source '{source}' and provider '{requested_provider}' "
                    f"available versions are {versions}"
                )

        return self._sources[source][requested_provider][requested_version]

    def add_source(self, name: str, source: DataSource) -> None:
        """Add a new data source to the data catalog.

        The data version and provider are extracted from the DataSource object.

        Parameters
        ----------
        source : str
            Name of the data source.
        adapter : DataAdapter
            DataAdapter object.
        """
        if not isinstance(source, DataSource):
            raise ValueError("Value must be DataSource")

        if source.version:
            version = str(source.version)
        else:
            version = "_UNSPECIFIED_"  # make sure this comes first in sorted list

        if source.provider:
            provider = str(source.provider)
        else:
            protocol = source.driver.filesystem.protocol
            if isinstance(protocol, str):
                provider: str = protocol
            else:
                provider: str = protocol[0]

        if name not in self._sources:
            self._sources[name] = {}
        else:  # check if data type is the same as source with same name
            source0 = next(iter(next(iter(self._sources[name].values())).values()))
            if source0.data_type != source.data_type:
                raise ValueError(
                    f"Data source '{name}' already exists with data type "
                    f"'{source0.data_type}' but new data source has data type "
                    f"'{source.data_type}'."
                )

        if provider not in self._sources[name]:
            versions = {str(version): source}
        else:
            versions = self._sources[name][provider]
            if provider in self._sources[name] and version in versions:
                warnings.warn(
                    f"overwriting data source '{name}' with "
                    f"provider {provider} and version {version}.",
                    UserWarning,
                    stacklevel=2,
                )
            # update and sort dictionary -> make sure newest version is last
            versions.update({str(version): source})
            versions = {(k): versions[k] for k in sorted(list(versions.keys()))}

        self._sources[name][provider] = versions

    def list_sources(self, used_only=False) -> List[Tuple[str, DataSource]]:
        """Return a flat list of all available data sources.

        Parameters
        ----------
        used_only: bool, optional
            If True, return only data entries marked as used, by default False.
        """
        sources = []
        for source_name, available_providers in self._sources.items():
            for _, available_versions in available_providers.items():
                for _, source in available_versions.items():
                    if used_only and not source._used:
                        continue
                    sources.append((source_name, source))

        return sources

    def __iter__(self) -> Iterator[Tuple[str, DataAdapter]]:
        """Iterate over sources."""
        return iter(self.list_sources())

    def contains_source(
        self,
        source: str,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        permissive: bool = True,
    ) -> bool:
        """
        Check if source is in catalog.

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
        permissive : bool, optional
            Whether variant checking is necessary. If true, the name of the source
            only is checked, if false, and at least one of version or provider is
            not None, this will only return True if that variant specifically is
            available.


        Returns
        -------
        bool
            whether the source (with specified variants if necessary) is available
        """
        if permissive or (version is None and provider is None):
            return source in self._sources
        else:
            if version:
                if version not in self._sources[source]:
                    return False
                else:
                    selected_version = version
            else:
                selected_version = next(iter(self._sources[source].keys()))

            return provider not in self._sources[source][selected_version].keys()

    def __len__(self):
        """Return number of sources."""
        return len(self.list_sources())

    def __repr__(self):
        """Prettyprint the sources."""
        return self.to_dataframe().__repr__()

    def __eq__(self, other) -> bool:
        if type(other) is type(self):
            if len(self) != len(other):
                return False
            for name, source in self.list_sources():
                try:
                    other_source = other.get_source(
                        name, provider=source.provider, version=str(source.version)
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

    def _set_predefined_catalogs(self) -> Dict:
        """Set initialized predefined catalogs to _catalogs attribute."""
        for k, cat in PLUGINS.catalog_plugins.items():
            self._catalogs[k] = cat(
                format_version=self._format_version, cache_dir=self._cache_dir
            )
        return self._catalogs

    def from_predefined_catalogs(self, name: str, version: str = "latest") -> None:
        """Add data sources from a predefined data catalog.

        Parameters
        ----------
        name : str
            Catalog name.
        version : str, optional
            Catlog release version. By default it takes the latest known release.

        """
        if "=" in name:
            name, version = name.split("=")[0], name.split("=")[-1]
        if name not in self.predefined_catalogs:
            raise ValueError(
                f'Catalog with name "{name}" not found in predefined catalogs'
            )
        # cache and get path to data_datalog.yml file of the <name> catalog with <version>
        catalog_path = self.predefined_catalogs[name].get_catalog_file(version)
        # read catalog
        self.logger.info(f"Reading data catalog {name} {version}")
        self.from_yml(catalog_path, catalog_name=name)

    def _cache_archive(
        self,
        archive_uri: str,
        name: str,
        version: str = "latest",
        sha256: Optional[str] = None,
    ) -> str:
        """Cache a data archive to the cache directory.

        The archive is unpacked and cached to <cache_dir>/<name>/<version>

        Parameters
        ----------
        archive_uri : str
            uri to data archive.
        name : str
            Name of data catalog
        version : str, optional
            Version of data archive, by default 'latest'.
        sha256 : str, optional
            SHA256 hash of the archive, by default None.

        Returns
        -------
        str
            Path to the datacatalog of the cached data archive

        """
        root = Path(self._cache_dir, name, version)
        # retrieve and unpack archive
        kwargs = {}
        if Path(archive_uri).suffix == ".zip":
            kwargs.update(processor=pooch.Unzip(extract_dir=root))
        elif Path(archive_uri).suffix == ".gz":
            kwargs.update(processor=pooch.Untar(extract_dir=root))
        if Path(archive_uri).exists():  # check if arhive is a local file
            kwargs.update(donwloader=_copy_file)
        pooch.retrieve(
            archive_uri,
            known_hash=sha256,
            path=str(root),
            fname=Path(archive_uri).name,
            **kwargs,
        )
        return root

    def from_yml(
        self,
        urlpath: Union[Path, str],
        root: Optional[StrPath] = None,
        catalog_name: Optional[str] = None,
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

        .. code-block:: yaml

            meta:
              root: <path>
              category: <category>
              version: <version>
              name: <name>
              sha256: <sha256> # only if the root is an archive
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
        # read meta data
        meta = yml.pop("meta", meta)
        if catalog_name is None:
            catalog_name = cast(
                str, meta.get("name", "".join(basename(urlpath).split(".")[:-1]))
            )
        version = meta.get("version", None)
        if root is None:
            root = meta.get("root", os.path.dirname(urlpath))
        if root.split(".")[-1] in ["gz", "zip"]:
            # if root is an archive, unpack it at the cache dir
            root = self._cache_archive(
                archive_uri=root,
                name=catalog_name,
                version=version,
                sha256=meta.get("sha256", None),
            )
        self.from_dict(
            yml,
            catalog_name=catalog_name,
            root=root,
            category=meta.get("category", None),
            mark_used=mark_used,
        )
        return self

    def _is_compatible(
        self, hydromt_version: str, requested_range: str, allow_prerelease=True
    ) -> bool:
        if requested_range is None:
            return True
        requested = SpecifierSet(requested_range)
        version = Version(hydromt_version)

        if allow_prerelease:
            return version in requested or Version(version.base_version) in requested
        else:
            return version in requested

    def _determine_catalog_root(
        self, meta: Dict[str, Any], urlpath: Optional[StrPath] = None
    ) -> Path:
        """Determine which of the roots provided in meta exists and should be used."""
        root = None
        for r in meta["roots"]:
            if exists(r):
                root = r
                break

        if root is None:
            raise ValueError("None of the specified roots were found")
        else:
            return Path(root)

    def from_dict(
        self,
        data_dict: Dict,
        catalog_name: str = "",
        root: Optional[StrPath] = None,
        category: Optional[str] = None,
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
        # check version required hydromt version
        requested_version = meta.get("hydromt_version", None)
        if requested_version is not None:
            allow_dev = meta.get("allow_dev_version", True)
            if not self._is_compatible(__version__, requested_version, allow_dev):
                raise RuntimeError(
                    f"Data catalog requires Hydromt Version {requested_version} which "
                    f"is incompattible with current hydromt version {__version__}."
                )

        if "category" in meta and category is None:
            category = meta.pop("category")
        version = meta.get("version", None)

        if root is not None:
            self.root = root
        elif "roots" in meta:
            self.root = self._determine_catalog_root(meta)

        self.logger.info(f"Data Catalog is using root: {self.root}")

        if self.root is not None and splitext(self.root)[-1] in ["gz", "zip"]:
            # if root is an archive, unpack it at the cache dir
            self.root = self._cache_archive(
                self.root, name=catalog_name, version=version
            )

            # save catalog to cache
            with open(join(self.root, f"{catalog_name}.yml"), "w") as f:
                d = {"meta": {k: v for k, v in meta.items() if k != "roots"}}
                d.update(data_dict)
                yaml.dump(d, f, default_flow_style=False, sort_keys=False)

        for name, source_dict in _denormalise_data_dict(data_dict):
            source = _parse_data_source_dict(
                name,
                source_dict,
                root=root,
                category=category,
            )
            if mark_used:
                source.mark_as_used()
            self.add_source(name, source)

        return self

    def to_yml(
        self,
        path: Union[str, Path],
        root: str = "auto",
        source_names: Optional[List] = None,
        used_only: bool = False,
        meta: Optional[Dict] = None,
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
        meta = meta or []
        yml_dir = os.path.dirname(abspath(path))
        if root == "auto":
            root = yml_dir
        data_dict = self.to_dict(
            root=root, source_names=source_names, meta=meta, used_only=used_only
        )
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
        meta: Optional[dict] = None,
        used_only: bool = False,
    ) -> Dict:
        """Export the data catalog to a dictionary.

        Parameters
        ----------
        source_names : list, optional
            List of source names to export, by default None in which case all sources
            are exported.
        root : str, Path, optional
            Global root for all relative paths in the file.
        meta: dict, optional
            key-value pairs to add to the data catalog meta section, such as 'version',
            by default empty.
        used_only: bool, optional
            If True, export only data entries marked as used, by default False.

        Returns
        -------
        dict
            data catalog dictionary
        """
        meta = meta or {}
        sources_out = dict()
        if root is None:
            root = str(self.root)
        meta.update(**{"root": root})
        sources = self.list_sources(used_only=used_only)
        sorted_sources = sorted(sources, key=lambda x: x[0])
        for name, source in sorted_sources:  # alphabetical order
            if source_names is not None and name not in source_names:
                continue
            source_dict = source.model_dump(
                exclude_defaults=True,  # keeps catalog as clean as possible
                exclude=["name"],  # name is already in the key
                round_trip=True,
            )

            # remove non serializable entries to prevent errors
            source_dict = _process_dict(source_dict, logger=self.logger)
            source_dict["root"] = root
            if name in sources_out:
                existing = sources_out.pop(name)
                if existing == source_dict:
                    sources_out.update({name: source_dict})
                    continue
                if "variants" in existing:
                    variants = existing.pop("variants")
                    _, variant, _ = _partition_dictionaries(source_dict, existing)
                    variants.append(variant)
                    existing["variants"] = variants
                else:
                    base, diff_existing, diff_new = _partition_dictionaries(
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

    def to_dataframe(self, source_names: Optional[List] = None) -> pd.DataFrame:
        """Return data catalog summary as DataFrame."""
        source_names = source_names or []
        d = []
        for name, source in self.list_sources():
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
        bbox: Optional[Bbox] = None,
        time_tuple: Optional[TimeRange] = None,
        source_names: Optional[List] = None,
        unit_conversion: bool = True,
        meta: Optional[Dict] = None,
        append: bool = False,
        handle_nodata: NoDataStrategy = NoDataStrategy.IGNORE,
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
        source_names = source_names or []
        meta = meta or {}
        data_root = abspath(data_root)
        if not os.path.isdir(data_root):
            os.makedirs(data_root)

        # create copy of data with selected source names
        source_vars = {}
        if len(source_names) > 0:
            sources = {}
            for source in source_names:
                # support both strings and SourceSpecDicts here
                if isinstance(source, str):
                    name = source
                elif isinstance(source, Dict):
                    name = source["source"]
                else:
                    raise RuntimeError(
                        f"unknown source type: {source} of type {type(source).__name__}"
                    )
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
                            unit_mult = source.data_adapter.unit_mult
                            unit_add = source.data_adapter.unit_add
                            source.data_adapter.unit_mult = {}
                            source.data_adapter.unit_add = {}
                        try:
                            fn_out, driver, driver_kwargs = source.to_file(
                                file_path=Path(data_root) / source.uri,
                                data_name=key,
                                variables=source_vars.get(key, None),
                                bbox=bbox,
                                time_tuple=time_tuple,
                                handle_nodata=NoDataStrategy.RAISE,
                                logger=self.logger,
                            )
                        except NoDataException as e:
                            exec_nodata_strat(
                                f"{key} file contains no data: {e}",
                                handle_nodata,
                                logger,
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
                        if driver_kwargs is not None:
                            source.driver_kwargs.update(driver_kwargs)
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
            for _provider, available_versions in available_variants.items():
                for _version, adapter in available_versions.items():
                    data_catalog_out.add_source(key, adapter)

        data_catalog_out.to_yml(fn, root="auto", meta=meta)

    def get_rasterdataset(
        self,
        data_like: Union[
            str, SourceSpecDict, Path, xr.Dataset, xr.DataArray, RasterDatasetSource
        ],
        bbox: Optional[List] = None,
        geom: Optional[gpd.GeoDataFrame] = None,
        zoom_level: Optional[Union[int, tuple]] = None,
        buffer: Union[float, int] = 0,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        variables: Optional[Union[List, str]] = None,
        time_tuple: Optional[Tuple] = None,
        single_var_as_array: Optional[bool] = True,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ) -> Optional[Union[xr.Dataset, xr.DataArray]]:
        """Return a clipped, sliced and unified RasterDataset.

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` argument.
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
        handle_nodata: NoDataStrategy, optional
            What to do if no data can be found.
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
            RasterDataset. If no data is found and handle_nodata is set to IGNORE None
            will be returned. if it is set to RAISE and exception will be raised in that
            situation
        """
        if isinstance(variables, str):
            variables = [variables]

        if isinstance(data_like, dict):
            data_like, provider, version = _parse_data_like_dict(
                data_like, provider, version
            )

        if isinstance(data_like, (str, Path)):
            if isinstance(data_like, str) and data_like in self.sources:
                name = data_like
                source = self.get_source(name, provider=provider, version=version)
            else:
                if "provider" not in kwargs:
                    kwargs.update({"provider": "user"})

                driver: str = kwargs.pop(
                    "driver", RasterDatasetSource._fallback_driver_read
                )
                name = basename(data_like)
                source = RasterDatasetSource(
                    name=name, uri=str(data_like), driver=driver
                )
                self.add_source(name, source)
        elif isinstance(data_like, (xr.DataArray, xr.Dataset)):
            if geom is not None or bbox is not None:
                mask = parse_geom_bbox_buffer(geom, bbox, buffer)
            else:
                mask = None
            data_like = RasterDatasetAdapter._slice_data(
                ds=data_like,
                variables=variables,
                mask=mask,
                time_tuple=time_tuple,
                logger=self.logger,
            )
            if data_like is None:
                exec_nodata_strat(
                    "No data was left after slicing.",
                    strategy=handle_nodata,
                    logger=logger,
                )
            ds = _single_var_as_array(
                maybe_ds=data_like,
                single_var_as_array=single_var_as_array,
                variable_name=variables,
            )
            return ds
        elif isinstance(data_like, RasterDatasetSource):
            source = data_like
        else:
            raise ValueError(f'Unknown raster data type "{type(data_like).__name__}"')

        obj = source.read_data(
            bbox=bbox,
            mask=geom,
            buffer=buffer,
            zoom_level=zoom_level,
            variables=variables,
            time_range=time_tuple,
            handle_nodata=handle_nodata,
            logger=self.logger,
        )
        obj = _single_var_as_array(
            maybe_ds=obj,
            single_var_as_array=single_var_as_array,
            variable_name=variables,
        )
        return obj

    def get_geodataframe(
        self,
        data_like: Union[
            str, SourceSpecDict, Path, xr.Dataset, xr.DataArray, GeoDataFrameSource
        ],
        bbox: Optional[List] = None,
        geom: Optional[gpd.GeoDataFrame] = None,
        buffer: Union[float, int] = 0,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        variables: Optional[Union[List, str]] = None,
        predicate: str = "intersects",
        provider: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ) -> Optional[gpd.GeoDataFrame]:
        """Return a clipped and unified GeoDataFrame (vector).

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` argument.
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
        handle_nodata : NoDataStrategy, optional
            How to handle no data values, by default NoDataStrategy.RAISE
        predicate : optional
            If predicate is provided, the GeoDataFrame is filtered by testing
            the predicate function against each item. Requires bbox or mask.
            By default 'intersects' options are:
            {'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches'},
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
        gdf: Optional[geopandas.GeoDataFrame]
            GeoDataFrame. If no data is found and handle_nodata is set to IGNORE None
            will be returned. if it is set to RAISE and exception will be raised in that
            situation
        """
        if geom is not None or bbox is not None:
            mask = parse_geom_bbox_buffer(geom=geom, bbox=bbox, buffer=buffer)
        else:
            mask = None
        if isinstance(data_like, dict):
            data_like, provider, version = _parse_data_like_dict(
                data_like, provider, version
            )
        if isinstance(data_like, (str, Path)):
            if str(data_like) in self.sources:
                name = str(data_like)
                source = self.get_source(name, provider=provider, version=version)
            else:
                if "provider" not in kwargs:
                    kwargs.update({"provider": "user"})
                driver: str = kwargs.pop(
                    "driver", GeoDataFrameSource._fallback_driver_read
                )
                name = basename(data_like)
                source = GeoDataFrameSource(
                    name=name, uri=str(data_like), driver=driver, **kwargs
                )
                self.add_source(name, source)
        elif isinstance(data_like, gpd.GeoDataFrame):
            data_like = GeoDataFrameAdapter._slice_data(
                data_like,
                variables=variables,
                mask=mask,
                predicate=predicate,
                logger=self.logger,
            )
            if data_like is None:
                exec_nodata_strat(
                    "No data was left after slicing.",
                    strategy=handle_nodata,
                    logger=logger,
                )
            return data_like
        elif isinstance(data_like, GeoDataFrameSource):
            source = data_like
        else:
            raise ValueError(f'Unknown vector data type "{type(data_like).__name__}"')

        gdf = source.read_data(
            mask=mask,
            handle_nodata=handle_nodata,
            predicate=predicate,
            variables=variables,
            logger=self.logger,
        )
        return gdf

    def get_geodataset(
        self,
        data_like: Union[
            str, SourceSpecDict, Path, xr.Dataset, xr.DataArray, GeoDatasetSource
        ],
        bbox: Optional[List] = None,
        geom: Optional[gpd.GeoDataFrame] = None,
        buffer: Union[float, int] = 0,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        predicate: str = "intersects",
        variables: Optional[List] = None,
        time_range: Optional[Union[Tuple[str, str], Tuple[datetime, datetime]]] = None,
        single_var_as_array: bool = True,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Return a clipped, sliced and unified GeoDataset.

        To clip the data to the area of interest, provide a `bbox` or `geom`,
        with optional additional `buffer` argument.
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
        handle_nodata: NoDataStrategy Optional
            what should happen if the requested data set is empty. RAISE by default
        predicate : optional
            If predicate is provided, the GeoDataFrame is filtered by testing
            the predicate function against each item. Requires bbox or mask.
            By default 'intersects' options are:
            {'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches'},
        variables : str or list of str, optional.
            Names of GeoDataset variables to return. By default all dataset variables
            are returned.
        time_range: tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        single_var_as_array: bool, optional
            If True, return a DataArray if the dataset consists of a single variable.
            If False, always return a Dataset. By default True.
        **kwargs:
            Additional keyword arguments that are passed to the `GeoDatasetSource`
            function. Only used if `data_like` is a path to a geodataset file.

        Returns
        -------
        obj: xarray.Dataset or xarray.DataArray
            GeoDataset
        """
        if geom is not None or bbox is not None:
            mask = parse_geom_bbox_buffer(geom=geom, bbox=bbox, buffer=buffer)
        else:
            mask = None
        if isinstance(data_like, dict):
            data_like, provider, version = _parse_data_like_dict(
                data_like, provider, version
            )
        if isinstance(data_like, (str, Path)):
            if isinstance(data_like, str) and data_like in self.sources:
                name = data_like
                source = self.get_source(name, provider=provider, version=version)
            else:
                if "provider" not in kwargs:
                    kwargs.update({"provider": "user"})
                driver: str = kwargs.pop(
                    "driver", GeoDatasetSource._fallback_driver_read
                )
                name = basename(data_like)
                source = GeoDatasetSource(
                    name=name,
                    uri=str(data_like),
                    driver=driver,
                    **kwargs,
                )
                self.add_source(name, source)
        elif isinstance(data_like, (xr.DataArray, xr.Dataset)):
            data_like = GeoDatasetAdapter._slice_data(
                data_like,
                variables=variables,
                mask=mask,
                predicate=predicate,
                time_range=time_range,
                logger=self.logger,
            )
            if data_like is None:
                exec_nodata_strat(
                    "No data was left after slicing.",
                    strategy=handle_nodata,
                    logger=logger,
                )
            return _single_var_as_array(data_like, single_var_as_array, variables)
        elif isinstance(data_like, GeoDatasetSource):
            source = data_like
        else:
            raise ValueError(f'Unknown geo data type "{type(data_like).__name__}"')

        obj = source.read_data(
            mask=mask,
            handle_nodata=handle_nodata,
            predicate=predicate,
            variables=variables,
            time_range=time_range,
            single_var_as_array=single_var_as_array,
        )
        return obj

    def get_dataset(
        self,
        data_like: Union[str, SourceSpecDict, Path, xr.Dataset, xr.DataArray],
        variables: Optional[List] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        time_tuple: Optional[Union[Tuple[str, str], Tuple[datetime, datetime]]] = None,
        single_var_as_array: bool = True,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Return a clipped, sliced and unified Dataset.

        To slice the data to the time period of interest, provide the
        `time_tuple` argument. To return only the dataset variables
        of interest provide the `variables` argument.

        NOTE: Unless `single_var_as_array` is set to False a single-variable data source
        will be returned as xarray.DataArray rather than a xarray.Dataset.

        Arguments
        ---------
        data_like: str, Path, xr.Dataset, xr.DataArray, SourceSpecDict
            Data catalog key, path to geodataset file or geodataset xarray object.
            The catalog key can be a string or a dictionary with the following keys:
            {'name', 'provider', 'version'}.
            If a path to a file is provided it will be added
            to the catalog with its based on the file basename.
        time_tuple : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        single_var_as_array: bool, optional
            If True, return a DataArray if the dataset consists of a single variable.
            If False, always return a Dataset. By default True.
        **kwargs:
            Additional keyword arguments that are passed to the `DatasetAdapter`
            function. Only used if `data_like` is a path to a geodataset file.

        Returns
        -------
        obj: xarray.Dataset or xarray.DataArray
            Dataset
        """
        if isinstance(data_like, dict):
            data_like, provider, version = _parse_data_like_dict(
                data_like, provider, version
            )
        if isinstance(data_like, (str, Path)):
            if isinstance(data_like, str) and data_like in self.sources:
                name = data_like
                source = self.get_source(name, provider=provider, version=version)
            else:
                if "provider" not in kwargs:
                    kwargs.update({"provider": "local"})
                source = DatasetAdapter(path=str(data_like), **kwargs)
                name = basename(data_like)
                self.add_source(name, source)

        elif isinstance(data_like, (xr.DataArray, xr.Dataset)):
            data_like = DatasetAdapter._slice_data(
                data_like,
                variables,
                time_tuple,
                logger=self.logger,
            )
            if data_like is None:
                exec_nodata_strat(
                    "No data was left after slicing.",
                    strategy=handle_nodata,
                    logger=logger,
                )
            return DatasetAdapter._single_var_as_array(
                data_like, single_var_as_array, variables
            )
        else:
            raise ValueError(f'Unknown data type "{type(data_like).__name__}"')

        obj = source.get_data(
            variables=variables,
            time_tuple=time_tuple,
            single_var_as_array=single_var_as_array,
            handle_nodata=handle_nodata,
        )
        return obj

    def get_dataframe(
        self,
        data_like: Union[str, SourceSpecDict, Path, pd.DataFrame, DataFrameSource],
        variables: Optional[List] = None,
        time_range: Optional[TimeRange] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
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
        time_range : tuple of str, datetime, optional
            Start and end date of period of interest. By default the entire time period
            of the dataset is returned.
        handle_nodata: NoDataStrategy Optional
            what should happen if the requested data set is empty. RAISE by default
        **kwargs:
            Additional keyword arguments that are passed to the `DataframeAdapter`
            function. Only used if `data_like` is a path to a tabular data file.

        Returns
        -------
        pd.DataFrame
            Tabular data. If no data is found and handle_nodata is set to IGNORE None
            will be returned. if it is set to RAISE and exception will be raised in that
            situation
        """
        if isinstance(data_like, dict):
            data_like, provider, version = _parse_data_like_dict(
                data_like, provider, version
            )
        if isinstance(data_like, (str, Path)):
            if isinstance(data_like, str) and data_like in self.sources:
                name = data_like
                source: DataSource = self.get_source(
                    name, provider=provider, version=version
                )
                if not isinstance(source, DataFrameSource):
                    raise ValueError(f"Source '{source.name}' is not a DataFrame.")
            else:
                if "provider" not in kwargs:
                    kwargs.update({"provider": "user"})
                driver: str = kwargs.pop(
                    "driver", DataFrameSource._fallback_driver_read
                )
                name = basename(data_like)
                source = DataFrameSource(
                    uri=data_like, name=name, driver=driver, **kwargs
                )
                self.add_source(name, source)
        elif isinstance(data_like, pd.DataFrame):
            df = DataFrameAdapter._slice_data(data_like, variables, time_range)
            if df is None:
                exec_nodata_strat(
                    "No data was left after slicing.",
                    strategy=handle_nodata,
                    logger=logger,
                )
            return df
        else:
            raise ValueError(f'Unknown tabular data type "{type(data_like).__name__}"')

        obj = source.read_data(
            variables=variables,
            time_range=time_range,
            handle_nodata=handle_nodata,
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
    root: Optional[Union[Path, str]] = None,
    category: Optional[str] = None,
) -> DataSource:
    """Parse data source dictionary."""
    # parse data
    source = data_source_dict.copy()  # important as we modify with pop

    source["name"] = name

    # add root
    if root:
        source.update({"root": str(root)})

    # source meta data
    meta: Dict[str, str] = source.get("metadata", {})
    if "category" not in meta and category is not None:
        meta.update(category=category)

    source["metadata"] = meta

    # driver arguments
    # driver_kwargs = source.pop("driver_kwargs", source.pop("kwargs", {}))
    # TODO: remove code under this depending on subclasses
    #       The DataCatalog should not have specific implementations for different drivers
    # for driver_kwarg in driver_kwargs:
    #     # required for geodataset where driver_kwargs can be a path
    #     if "fn" in driver_kwarg:
    #         driver_kwargs.update(
    #             {driver_kwarg: abs_path(root, driver_kwargs[driver_kwarg])}
    #         )
    # source["driver_kwargs"] = driver_kwargs

    return create_source(source)


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

    Expand possible versions and variants in data_dict.
    """
    data_list = []
    for name, source in data_dict.items():
        source = copy.deepcopy(source)
        data_dicts = []
        if "variants" in source:
            variants = source.pop("variants")
            for diff in variants:
                source_copy = copy.deepcopy(source)
                source_copy = {
                    str(k): v for (k, v) in deep_merge(source_copy, diff).items()
                }

                data_dicts.append({name: source_copy})
        elif "placeholders" in source:
            options = source.pop("placeholders")
            for combination in itertools.product(*options.values()):
                source_copy = copy.deepcopy(source)
                name_copy = name
                for k, v in zip(options.keys(), combination):
                    name_copy = name_copy.replace("{" + k + "}", v)
                    # TODO: seems like the job for a MetaDataResolver?
                    source_copy["uri"] = source_copy["uri"].replace("{" + k + "}", v)
                data_dicts.append({name_copy: source_copy})
        else:
            for k, v in source.items():
                if isinstance(v, (int, float)):
                    # numbers are pretty much always a version here,
                    # and we need strings, so just cast to string when
                    # we encoutner a number. not the pretties,
                    # but it will have to do for now.
                    source[k] = str(v)
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
