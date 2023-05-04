# -*- coding: utf-8 -*-
"""General and basic API for models in HydroMT"""

from abc import ABCMeta
import os
import glob
from os.path import join, isdir, isfile, abspath, dirname, basename, isabs
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import box
import logging
from pathlib import Path
import inspect
import warnings
from pyproj import CRS
import typing
from typing import Any, Dict, List, Tuple, Union, Optional

from ..data_catalog import DataCatalog
from .. import config, log, workflows
from ..raster import GEO_MAP_COORD

__all__ = ["Model"]

logger = logging.getLogger(__name__)


class Model(object, metaclass=ABCMeta):
    """General and basic API for models in HydroMT"""

    # FIXME
    _DATADIR = ""  # path to the model data folder
    _NAME = "modelname"
    _CONF = "model.ini"
    _CF = dict()  # configreader kwargs
    _GEOMS = {"<general_hydromt_name>": "<model_name>"}
    _MAPS = {"<general_hydromt_name>": "<model_name>"}
    _FOLDERS = [""]
    # tell hydroMT which methods should receive the res and region arguments
    # TODO: change it back to setup_region and no res --> deprecation
    _CLI_ARGS = {"region": "setup_basemaps", "res": "setup_basemaps"}

    _API = {
        "crs": CRS,
        "config": Dict[str, Any],
        "geoms": Dict[str, gpd.GeoDataFrame],
        "maps": Dict[str, Union[xr.DataArray, xr.Dataset]],
        "forcing": Dict[str, Union[xr.DataArray, xr.Dataset]],
        "region": gpd.GeoDataFrame,
        "results": Dict[str, Union[xr.DataArray, xr.Dataset]],
        "states": Dict[str, Union[xr.DataArray, xr.Dataset]],
    }

    def __init__(
        self,
        root: Optional[str] = None,
        mode: Optional[str] = "w",
        config_fn: Optional[str] = None,
        data_libs: Union[List, str] = [],
        logger=logger,
        **artifact_keys,
    ):
        """Initialize a model

        Parameters
        ----------
        root : str, optional
            Model root, by default None
        mode : {'r','r+','w'}, optional
            read/append/write mode, by default "w"
        config_fn : str, optional
            Model simulation configuration file, by default None.
            Note that this is not the HydroMT model setup configuration file!
        data_libs : List[str], optional
            List of data catalog yaml files, by default None
        """
        from . import MODELS  # avoid circular import

        self.logger = logger
        dist, version = "unknown", "NA"
        if self._NAME in MODELS:
            ep = MODELS[self._NAME]
            dist, version = ep.distro.name, ep.distro.version

        # link to data
        self.data_catalog = DataCatalog(
            data_libs=data_libs, logger=self.logger, **artifact_keys
        )

        # placeholders
        # metadata maps that can be at different resolutions #TODO> do we want read/write maps?
        self._config = dict()  # nested dictionary
        self._maps = dict()  # dictionary of xr.DataArray and/or xr.Dataset
        # NOTE was staticgeoms in <=v0.5
        self._geoms = dict()  # dictionary of gdp.GeoDataFrame
        self._forcing = dict()  # dictionary of xr.DataArray and/or xr.Dataset
        self._states = dict()  # dictionary of xr.DataArray and/or xr.Dataset
        self._results = dict()  # dictionary of xr.DataArray and/or xr.Dataset
        # To be deprecated in future versions!
        self._staticmaps = xr.Dataset()
        self._staticgeoms = dict()

        # file system
        self._root = ""
        self._read = True
        self._write = False

        # model paths
        self._config_fn = self._CONF if config_fn is None else config_fn
        self.set_root(root, mode)  # also creates hydromt.log file
        self.logger.info(f"Initializing {self._NAME} model from {dist} (v{version}).")

    @property
    def api(self) -> Dict:
        """Return all model components and their data types"""
        _api = self._API.copy()
        # loop over parent and mixin classes and update API
        for base_cls in self.__class__.__bases__:
            _api.update(getattr(base_cls, "_API", {}))
        return _api

    def _check_get_opt(self, opt):
        """Check all opt keys and raise sensible error messages if unknown."""
        for method in opt.keys():
            m = method.strip("0123456789")
            if not callable(getattr(self, m, None)):
                raise ValueError(f'Model {self._NAME} has no method "{method}"')
        return opt

    def _run_log_method(self, method, *args, **kwargs):
        """Log method parameters before running a method"""
        method = method.strip("0123456789")
        func = getattr(self, method)
        signature = inspect.signature(func)
        # combine user and default options
        params = {}
        for i, (k, v) in enumerate(signature.parameters.items()):
            if k in ["args", "kwargs"]:
                if k == "args":
                    params[k] = args[i:]
                else:
                    params.update(**kwargs)
            else:
                v = kwargs.get(k, v.default)
                if len(args) > i:
                    v = args[i]
                params[k] = v
        # log options
        for k, v in params.items():
            if v is not inspect._empty:
                self.logger.info(f"{method}.{k}: {v}")
        return func(*args, **kwargs)

    def build(
        self,
        region: Optional[dict] = None,
        write: Optional[bool] = True,
        opt: Optional[dict] = {},
    ):
        """Single method to build a model from scratch based on settings in `opt`.

        Methods will be run one by one based on the order of appearance in `opt` (.ini configuration file).
        All model methods are supported including setup_*, read_* and write_* methods.

        If a write_* option is listed in `opt` (ini file) the full writing of the model at the end
        of the update process is skipped.

        Parameters
        ----------
        region: dict
            Description of model region. See :py:meth:`~hydromt.workflows.parse_region`
            for all options.
        write: bool, optional
            Write the complete model after executing all methods in opt, by default True.
        opt: dict, optional
            Model build configuration. The configuration can be parsed from a
            .ini file using :py:meth:`~hydromt.config.configread`.
            This is a nested dictionary where the first-level keys are the names of model
            specific (setup) methods and the second-level contain argument-value pairs of the method.

            .. code-block:: text

                {
                    <name of method1>: {
                        <argument1>: <value1>, <argument2>: <value2>
                    },
                    <name of method2>: {
                        ...
                    }
                }

        """
        opt = self._check_get_opt(opt)

        # merge cli region and res arguments with opt
        if region is not None:
            if self._CLI_ARGS["region"] not in opt:
                opt = {self._CLI_ARGS["region"]: {}, **opt}
            opt[self._CLI_ARGS["region"]].update(region=region)

        # then loop over other methods
        for method in opt:
            # if any write_* functions are present in opt, skip the final self.write() call
            if method.startswith("write_"):
                write = False
            kwargs = {} if opt[method] is None else opt[method]
            self._run_log_method(method, **kwargs)

        # write
        if write:
            self.write()

    def update(
        self,
        model_out: Optional[Union[str, Path]] = None,
        write: Optional[bool] = True,
        opt: Dict = {},
    ):
        """Single method to update a model based the settings in `opt`.

        Methods will be run one by one based on the order of appearance in `opt` (ini configuration file).

        All model methods are supported including setup_*, read_* and write_* methods.
        If a write_* option is listed in `opt` (ini file) the full writing of the model at the end
        of the update process is skipped.

        Parameters
        ----------
        model_out: str, path, optional
            Destination folder to write the model schematization after updating
            the model. If None the updated model components are overwritten in the
            current model schematization if these exist. By default None.
        write: bool, optional
            Write the updated model schematization to disk. By default True.
        opt: dict, optional
            Model build configuration. The configuration can be parsed from a
            .ini file using :py:meth:`~hydromt.config.configread`.
            This is a nested dictionary where the first-level keys are the names of model
            specific (setup) methods and the second-level contain argument-value pairs of the method.

            .. code-block:: text

                {
                    <name of method1>: {
                        <argument1>: <value1>, <argument2>: <value2>
                    },
                    <name of method2>: {
                        ...
                    }
                }
        """
        opt = self._check_get_opt(opt)

        # read current model
        if not self._write:
            if model_out is None:
                raise ValueError(
                    '"model_out" directory required when updating in "read-only" mode'
                )
            self.read()
            self.set_root(model_out, mode="w")

        # check if model has a region
        if self.region is None:
            raise ValueError("Model region not found, setup model using `build` first.")

        # remove setup_basemaps from options and throw warning
        method = self._CLI_ARGS["region"]
        if method in opt:
            opt.pop(method)  # remove from opt
            self.logger.warning(f'"{method}" can only be called when building a model.')

        # loop over other methods from ini file
        for method in opt:
            # if any write_* functions are present in opt, skip the final self.write() call
            if method.startswith("write_"):
                write = False
            kwargs = {} if opt[method] is None else opt[method]
            self._run_log_method(method, **kwargs)

        # write
        if write:
            self.write()

    ## general setup methods

    def setup_region(
        self,
        region: dict,
        hydrography_fn: str = "merit_hydro",
        basin_index_fn: str = "merit_hydro_index",
    ) -> dict:
        """
        This component sets the `region` of interest of the model.

        Adds model layer:

        * **region** geom: region boundary vector

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g.:

            * {'bbox': [xmin, ymin, xmax, ymax]}

            * {'geom': 'path/to/polygon_geometry'}

            * {'basin': [xmin, ymin, xmax, ymax]}

            * {'subbasin': [x, y], '<variable>': threshold}

            For a complete overview of all region options,
            see :py:function:~hydromt.workflows.basin_mask.parse_region
        hydrography_fn : str
            Name of data source for hydrography data.
            FIXME describe data requirements
        basin_index_fn : str
            Name of data source with basin (bounding box) geometries associated with
            the 'basins' layer of `hydrography_fn`. Only required if the `region` is
            based on a (sub)(inter)basins without a 'bounds' argument.

        Returns
        -------
        region: dict
            Parsed region dictionary

        See Also
        --------
        hydromt.workflows.basin_mask.parse_region
        """
        kind, region = workflows.parse_region(region, logger=self.logger)
        # NOTE: kind=outlet is deprecated!
        if kind in ["basin", "subbasin", "interbasin", "outlet"]:
            # retrieve global hydrography data (lazy!)
            ds_org = self.data_catalog.get_rasterdataset(hydrography_fn)
            if "bounds" not in region:
                region.update(basin_index=self.data_catalog[basin_index_fn])
            # get basin geometry
            geom, xy = workflows.get_basin_geometry(
                ds=ds_org,
                kind=kind,
                logger=self.logger,
                **region,
            )
            region.update(xy=xy)
        elif "bbox" in region:
            bbox = region["bbox"]
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
        elif "geom" in region:
            geom = region["geom"]
            if geom.crs is None:
                raise ValueError('Model region "geom" has no CRS')
        elif "grid" in region:  # Grid specific - should be removed in the future
            geom = region["grid"].raster.box
        elif "model" in region:
            geom = region["model"].region
        else:
            raise ValueError(f"model region argument not understood: {region}")

        self.set_geoms(geom, name="region")

        # This setup method returns region so that it can be wrapped for models which require
        # more information, e.g. grid RasterDataArray or xy coordinates.
        return region

    # TODO remove
    # placeholder to make make sure build with the current _CLI_ARGS does not raise an error
    def setup_basemaps(self, *args, **kwargs):
        warnings.warn(
            "The setup_basemaps method is not implemented.",
            UserWarning,
        )

    ## file system

    @property
    def root(self):
        """Path to model folder."""
        if self._root is None:
            raise ValueError("Root unknown, use set_root method")
        return self._root

    @property
    def _assert_write_mode(self):
        if not self._write:
            raise IOError("Model opened in read-only mode")

    @property
    def _assert_read_mode(self):
        if not self._read:
            raise IOError("Model opened in write-only mode")

    def set_root(self, root: Optional[str], mode: Optional[str] = "w"):
        """Initialize the model root.

        In read/append mode a check is done if the root exists.
        In write mode the required model folder structure is created.

        Parameters
        ----------
        root : str, optional
            path to model root
        mode : {"r", "r+", "w"}, optional
            read/append/write mode for model files
        """
        ignore_ext = set([".log", ".yml"])
        if mode not in ["r", "r+", "w", "w+"]:
            raise ValueError(
                f'mode "{mode}" unknown, select from "r", "r+", "w" or "w+"'
            )
        # old_root = getattr(self, "_root", None)
        self._root = root if root is None else abspath(root)
        self._read = mode.startswith("r")
        self._write = mode != "r"
        self._overwrite = mode == "w+"
        if root is not None:
            if self._write:
                for name in self._FOLDERS:
                    path = join(self._root, name)
                    if not isdir(path):
                        os.makedirs(path)
                        continue
                    # path already exists check files
                    fns = glob.glob(join(path, "*.*"))
                    exts = set([os.path.splitext(fn)[1] for fn in fns])
                    exts -= ignore_ext
                    if len(exts) != 0:
                        if mode.endswith("+"):
                            self.logger.warning(
                                "Model dir already exists and "
                                f"files might be overwritten: {path}."
                            )
                        else:
                            msg = (
                                f"Model dir already exists and cannot be overwritten: {path}."
                                "Use 'mode=w+' to force overwrite existing files."
                            )
                            self.logger.error(msg)
                            raise IOError(msg)
            # check directory
            elif not isdir(self._root):
                raise IOError(f'model root not found at "{self._root}"')
            # remove old logging file handler and add new filehandler in root if it does not exist
            has_log_file = False
            log_level = 20  # default, but overwritten by the level of active loggers
            for i, h in enumerate(self.logger.handlers):
                log_level = h.level
                if hasattr(h, "baseFilename"):
                    if dirname(h.baseFilename) != self._root:
                        self.logger.handlers.pop(
                            i
                        ).close()  # remove handler and close file
                    else:
                        has_log_file = True
                    break
            if not has_log_file:
                new_path = join(self._root, "hydromt.log")
                log.add_filehandler(self.logger, new_path, log_level)

    # I/O
    def read(
        self,
        components: List = [
            "config",
            "staticmaps",
            "maps",
            "geoms",
            "forcing",
            "states",
            "results",
        ],
    ) -> None:
        """Read the complete model schematization and configuration from model files.

        Parameters
        ----------
        components : List, optional
            List of model components to read, each should have an associated read_<component> method.
            By default ['config', 'maps', 'staticmaps', 'geoms', 'forcing', 'states', 'results']
        """
        self.logger.info(f"Reading model data from {self.root}")
        for component in components:
            if not hasattr(self, f"read_{component}"):
                raise AttributeError(
                    f"{type(self).__name__} does not have read_{component}"
                )
            getattr(self, f"read_{component}")()

    def write(
        self,
        components: List = [
            "staticmaps",
            "maps",
            "geoms",
            "forcing",
            "states",
            "config",
        ],
    ) -> None:
        """Write the complete model schematization and configuration to model files.

        Parameters
        ----------
        components : List, optional
            List of model components to write, each should have an associated write_<component> method.
            By default ['config', 'maps', 'staticmaps', 'geoms', 'forcing', 'states']
        """
        self.logger.info(f"Writing model data to {self.root}")
        for component in components:
            if not hasattr(self, f"write_{component}"):
                raise AttributeError(
                    f"{type(self).__name__} does not have write_{component}"
                )
            getattr(self, f"write_{component}")()

    def write_data_catalog(
        self,
        root: Optional[Union[str, Path]] = None,
        data_lib_fn: Union[str, Path] = "hydromt_data.yml",
        used_only: bool = True,
        append: bool = True,
    ):
        """Write the data catalog to data_lib_fn

        Parameters
        ----------
        root: str, Path, optional
            Global root for all relative paths in yaml file.
            If "auto" the data source paths are relative to the yaml output ``path``.
        data_lib_fn: str, Path, optional
            Path of output yml file, absolute or relative to the model root, by default "hydromt_data.yml".
        used_only: bool, optional
            If True, export only data entries kept in used_data list. By default True
        append: bool, optional
            If True, append to an existing
        """
        path = data_lib_fn if isabs(data_lib_fn) else join(self.root, data_lib_fn)
        cat = DataCatalog(logger=self.logger, fallback_lib=None)
        # read hydromt_data yaml file and add to data catalog
        if self._read and isfile(path) and append:
            cat.from_yml(path)
        # update data catalog with new used sources
        source_names = (
            self.data_catalog._used_data
            if used_only
            else list(self.data_catalog.sources.keys())
        )
        if len(source_names) > 0:
            cat.from_dict(self.data_catalog.to_dict(source_names=source_names))
        if cat.sources:
            self._assert_write_mode
            cat.to_yml(path, root=root)

    # model configuration
    @property
    def config(self) -> Dict[str, Union[Dict, str]]:
        """Model configuration. Returns a (nested) dictionary"""
        # initialize default config if in write-mode
        if not self._config:
            self.read_config()
        return self._config

    def set_config(self, *args):
        """Update the config dictionary at key(s) with values.

        Parameters
        ----------
        args : key(s), value tuple, with minimal length of two
            keys can given by multiple args: ('key1', 'key2', 'value')
            or a string with '.' indicating a new level: ('key1.key2', 'value')

        Examples
        --------
        >> # self.config = {'a': 1, 'b': {'c': {'d': 2}}}

        >> set_config('a', 99)
        >> {'a': 99, 'b': {'c': {'d': 2}}}

        >> set_config('b', 'c', 'd', 99) # identical to set_config('b.d.e', 99)
        >> {'a': 1, 'b': {'c': {'d': 99}}}
        """
        if len(args) < 2:
            raise TypeError("set_config() requires a least one key and one value.")
        args = list(args)
        value = args.pop(-1)
        if len(args) == 1 and "." in args[0]:
            args = args[0].split(".") + args[1:]
        branch = self.config  # reads config at first call
        for key in args[:-1]:
            if not key in branch or not isinstance(branch[key], dict):
                branch[key] = {}
            branch = branch[key]
        branch[args[-1]] = value

    def setup_config(self, **cfdict):
        """Update config with a dictionary"""
        # TODO rename to update_config
        if len(cfdict) > 0:
            self.logger.debug(f"Setting model config options.")
        for key, value in cfdict.items():
            self.set_config(key, value)

    def get_config(self, *args, fallback=None, abs_path: Optional[bool] = False):
        """Get a config value at key(s).

        Parameters
        ----------
        args : tuple or string
            keys can given by multiple args: ('key1', 'key2')
            or a string with '.' indicating a new level: ('key1.key2')
        fallback: any, optional
            fallback value if key(s) not found in config, by default None.
        abs_path: bool, optional
            If True return the absolute path relative to the model root, by deafult False.
            NOTE: this assumes the config is located in model root!

        Returns
        value : any type
            dictionary value

        Examples
        --------
        >> # self.config = {'a': 1, 'b': {'c': {'d': 2}}}

        >> get_config('a')
        >> 1

        >> get_config('b', 'c', 'd') # identical to get_config('b.c.d')
        >> 2

        >> get_config('b.c') # # identical to get_config('b','c')
        >> {'d': 2}
        """
        args = list(args)
        if len(args) == 1 and "." in args[0]:
            args = args[0].split(".") + args[1:]
        branch = self.config  # reads config at first call
        for key in args[:-1]:
            branch = branch.get(key, {})
            if not isinstance(branch, dict):
                branch = dict()
                break
        value = branch.get(args[-1], fallback)
        if abs_path and isinstance(value, str):
            value = Path(value)
            if not value.is_absolute():
                value = Path(abspath(join(self.root, value)))
        return value

    def _configread(self, fn: str):
        return config.configread(fn, abs_path=False)

    def _configwrite(self, fn: str):
        return config.configwrite(fn, self.config)

    def read_config(self, config_fn: Optional[str] = None):
        """Parse config from file. If no config file found a default config file is
        read in writing mode."""
        prefix = "User defined"
        if config_fn is None:  # prioritize user defined config path (new v0.4.1)
            if not self._read:  # write-only mode > read default config
                config_fn = join(self._DATADIR, self._NAME, self._CONF)
                prefix = "Default"
            elif self.root is not None:  # append or write mode > read model config
                config_fn = join(self.root, self._config_fn)
                prefix = "Model"
        cfdict = dict()
        if config_fn is not None:
            if isfile(config_fn):
                cfdict = self._configread(config_fn)
                self.logger.debug(f"{prefix} config read from {config_fn}")
            elif not self._read and prefix != "Default":  # skip for missing default
                self.logger.error(f"{prefix} config file not found at {config_fn}")
        self._config = cfdict

    def write_config(
        self, config_name: Optional[str] = None, config_root: Optional[str] = None
    ):
        """Write config to <root/config_fn>"""
        self._assert_write_mode
        if config_name is not None:
            self._config_fn = config_name
        elif self._config_fn is None:
            self._config_fn = self._CONF
        if config_root is None:
            config_root = self.root
        fn = join(config_root, self._config_fn)
        self.logger.info(f"Writing model config to {fn}")
        self._configwrite(fn)

    # model static maps
    @property
    def staticmaps(self):
        """Model static maps. Returns xarray.Dataset,
        ..NOTE: will be deprecated in future versions and replaced by `grid`
        """
        warnings.warn(
            "The staticmaps property of the Model class will be deprecated in future versions, "
            "use the grid property of the GridModel class instead.",
            DeprecationWarning,
        )
        if len(self._staticmaps) == 0 and self._read:
            self.read_staticmaps()
        return self._staticmaps

    def set_staticmaps(
        self, data: Union[xr.DataArray, xr.Dataset], name: Optional[str] = None
    ):
        """
        This method will be deprecated in future versions. See :py:meth:`~hydromt.models.GridModel.set_grid`

        Add data to staticmaps.

        All layers of staticmaps must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to staticmaps
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        """
        warnings.warn(
            "The set_staticmaps method will be deprecated in future versions, use set_grid instead.",
            DeprecationWarning,
        )
        if name is None:
            if isinstance(data, xr.DataArray) and data.name is not None:
                name = data.name
            elif not isinstance(data, xr.Dataset):
                raise ValueError("Setting a map requires a name")
        elif name is not None and isinstance(data, xr.Dataset):
            data_vars = list(data.data_vars)
            if len(data_vars) == 1 and name not in data_vars:
                data = data.rename_vars({data_vars[0]: name})
            elif name not in data_vars:
                raise ValueError("Name not found in DataSet")
            else:
                data = data[[name]]
        if isinstance(data, xr.DataArray):
            data.name = name
            data = data.to_dataset()
        if len(self._staticmaps) == 0:  # new data
            self._staticmaps = data
        else:
            if isinstance(data, np.ndarray):
                if data.shape != self.shape:
                    raise ValueError("Shape of data and staticmaps do not match")
                data = xr.DataArray(dims=self.dims, data=data, name=name).to_dataset()
            for dvar in data.data_vars.keys():
                if dvar in self._staticmaps:
                    self.logger.warning(f"Replacing staticmap: {dvar}")
                self._staticmaps[dvar] = data[dvar]

    def read_staticmaps(self, fn: str = "staticmaps/staticmaps.nc", **kwargs) -> None:
        """Read static model maps at <root>/<fn> and add to staticmaps property

        key-word arguments are passed to :py:func:`xarray.open_dataset`

        .. NOTE: this method is deprecated. Use the grid property of the GridMixin instead.

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default "staticmaps/staticmaps.nc"
        """
        self._assert_read_mode
        for ds in self._read_nc(fn, **kwargs).values():
            self.set_staticmaps(ds)

    def write_staticmaps(self, fn: str = "staticmaps/staticmaps.nc", **kwargs) -> None:
        """Write static model maps to netcdf file at <root>/<fn>

        key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

        .. NOTE: this method is deprecated. Use the grid property of the GridMixin instead.

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, by default 'staticmaps/staticmaps.nc'
        """
        if len(self._staticmaps) == 0:
            self.logger.debug("No staticmaps data found, skip writing.")
        else:
            self._assert_write_mode
            # _write_nc requires dict - use dummy 'staticmaps' key
            nc_dict = {"staticmaps": self._staticmaps}
            self._write_nc(nc_dict, fn, **kwargs)

    # map files setup methods
    def setup_maps_from_raster(
        self,
        raster_fn: Union[str, Path, xr.Dataset],
        variables: Optional[List] = None,
        fill_method: Optional[str] = None,
        name: Optional[str] = None,
        reproject_method: Optional[str] = None,
        split_dataset: Optional[bool] = True,
    ) -> List[str]:
        """
        This component adds data variable(s) from ``raster_fn`` to maps object.

        If raster is a dataset, all variables will be added unless ``variables`` list is specified.

        Adds model layers:

        * **raster.name** maps: data from raster_fn

        Parameters
        ----------
        raster_fn: str, Path, xr.Dataset
            Data catalog key, path to raster file or raster xarray data object.
        variables: list, optional
            List of variables to add to maps from raster_fn. By default all.
        fill_method : str, optional
            If specified, fills nodata values using fill_nodata method.
            Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        name: str, optional
            Name of new maps variable, only in case split_dataset=False.
        reproject_method: str, optional
            See rasterio.warp.reproject for existing methods, by default the data is not reprojected (None).
        split_dataset: bool, optional
            If data is a xarray.Dataset split it into several xarray.DataArrays (default).

        Returns
        -------
        list
            Names of added model map layers
        """
        self.logger.info(f"Preparing maps data from raster source {raster_fn}")
        # Read raster data and select variables
        ds = self.data_catalog.get_rasterdataset(
            raster_fn,
            geom=self.region,
            buffer=2,
            variables=variables,
            single_var_as_array=False,
        )
        # Fill nodata
        if fill_method is not None:
            ds = ds.raster.interpolate_na(method=fill_method)
        # Reprojection
        if ds.rio.crs != self.crs and reproject_method is not None:
            ds = ds.raster.reproject(dst_crs=self.crs, method=reproject_method)
        # Add to maps
        self.set_maps(ds, name=name, split_dataset=split_dataset)

        return list(ds.data_vars.keys())

    def setup_maps_from_raster_reclass(
        self,
        raster_fn: Union[str, Path, xr.DataArray],
        reclass_table_fn: Union[str, Path, pd.DataFrame],
        reclass_variables: List,
        variable: Optional[str] = None,
        fill_method: Optional[str] = None,
        reproject_method: Optional[str] = None,
        name: Optional[str] = None,
        split_dataset: Optional[bool] = True,
        **kwargs,
    ) -> List[str]:
        """
        This component adds data variable(s) to maps object by reclassifying the data in ``raster_fn`` based on ``reclass_table_fn``.

        Adds model layers:

        * **reclass_variables** maps: reclassified raster data

        Parameters
        ----------
        raster_fn: str, Path, xr.DataArray
            Data catalog key, path to raster file or raster xarray data object. Should be a DataArray. Else use `variable` argument for selection.
        reclass_table_fn: str, Path, pd.DataFrame
            Data catalog key, path to tabular data file or tabular pandas dataframe object for the reclassification table of `raster_fn`.
        reclass_variables: list
            List of reclass_variables from reclass_table_fn table to add to maps. Index column should match values in `raster_fn`.
        variable: str, optional
            Name of raster dataset variable to use. This is only required when reading datasets with multiple variables.
            By default None.
        fill_method : str, optional
            If specified, fills nodata values in `raster_fn` using fill_nodata method before reclassifying.
            Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        reproject_method: str, optional
            See rasterio.warp.reproject for existing methods, by default the data is not reprojected (None).
        name: str, optional
            Name of new maps variable, only in case split_dataset=False.
        split_dataset: bool, optional
            If data is a xarray.Dataset split it into several xarray.DataArrays (default).

        Returns
        -------
        list
            Names of added model map layers
        """
        self.logger.info(
            f"Preparing map data by reclassifying the data in {raster_fn} based on {reclass_table_fn}"
        )
        # Read raster data and remapping table
        da = self.data_catalog.get_rasterdataset(
            raster_fn, geom=self.region, buffer=2, variables=variable, **kwargs
        )
        if not isinstance(da, xr.DataArray):
            raise ValueError(
                f"raster_fn {raster_fn} should be a single variable. "
                "Please select one using the 'variable' argument"
            )
        df_vars = self.data_catalog.get_dataframe(
            reclass_table_fn, variables=reclass_variables
        )
        # Fill nodata
        if fill_method is not None:
            da = da.raster.interpolate_na(method=fill_method)
        # Mapping function
        ds_vars = da.raster.reclassify(reclass_table=df_vars, method="exact")
        # Reprojection
        if ds_vars.rio.crs != self.crs and reproject_method is not None:
            ds_vars = ds_vars.raster.reproject(dst_crs=self.crs)
        # Add to maps
        self.set_maps(ds_vars, name=name, split_dataset=split_dataset)

        return list(ds_vars.data_vars.keys())

    # model map
    @property
    def maps(self) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """Model maps. Returns dict of xarray.DataArray or xarray.Dataset"""
        if len(self._maps) == 0 and self._read:
            self.read_maps()
        return self._maps

    def set_maps(
        self,
        data: Union[xr.DataArray, xr.Dataset],
        name: Optional[str] = None,
        split_dataset: Optional[bool] = True,
    ) -> None:
        """Add raster data to the maps component.

        Dataset can either be added as is (default) or split into several
        DataArrays using the split_dataset argument.

        Arguments
        ---------
        data: xarray.Dataset or xarray.DataArray
            New forcing data to add
        name: str, optional
            Variable name, only in case data is of type DataArray or if a Dataset is added as is (split_dataset=False).
        split_dataset: bool, optional
            If data is a xarray.Dataset split it into several xarray.DataArrays (default).
        """
        data_dict = _check_data(data, name, split_dataset)
        for name in data_dict:
            if name in self._maps:
                self.logger.warning(f"Replacing result: {name}")
            self._maps[name] = data_dict[name]

    def read_maps(self, fn: str = "maps/*.nc", **kwargs) -> None:
        """Read model map at <root>/<fn> and add to maps component

        key-word arguments are passed to :py:func:`xarray.open_dataset`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may wildcards, by default "maps/*.nc"
        """
        self._assert_read_mode
        ncs = self._read_nc(fn, **kwargs)
        for name, ds in ncs.items():
            self.set_maps(ds, name=name)

    def write_maps(self, fn="maps/{name}.nc", **kwargs) -> None:
        """Write maps to netcdf file at <root>/<fn>

        key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'maps/{name}.nc'
        """
        if len(self._maps) == 0:
            self.logger.debug("No maps data found, skip writing.")
        else:
            self._assert_write_mode
            self._write_nc(self._maps, fn, **kwargs)

    # model geometry files
    @property
    def geoms(self) -> Dict[str, Union[gpd.GeoDataFrame, gpd.GeoSeries]]:
        """Model geometries. Returns dict of geopandas.GeoDataFrame or geopandas.GeoDataSeries
        ..NOTE: previously call staticgeoms."""
        if not self._geoms and self._read:
            self.read_geoms()
        return self._geoms

    def set_geoms(self, geom: Union[gpd.GeoDataFrame, gpd.GeoSeries], name: str):
        """Add data to the geoms attribute.

        Arguments
        ---------
        geoms: geopandas.GeoDataFrame or geopandas.GeoSeries
            New geometry data to add
        name: str
            Geometry name.
        """
        gtypes = [gpd.GeoDataFrame, gpd.GeoSeries]
        if not np.any([isinstance(geom, t) for t in gtypes]):
            raise ValueError(
                "First parameter map(s) should be geopandas.GeoDataFrame or geopandas.GeoSeries"
            )
        if name in self._geoms:
            self.logger.warning(f"Replacing geom: {name}")
        self._geoms[name] = geom

    def read_geoms(self, fn: str = "geoms/*.geojson", **kwargs) -> None:
        """Read model geometries files at <root>/<fn> and add to geoms property

        key-word arguments are passed to :py:func:`geopandas.read_file`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may wildcards, by default "geoms/*.nc"
        """
        self._assert_read_mode
        fns = glob.glob(join(self.root, fn))
        for fn in fns:
            name = basename(fn).split(".")[0]
            self.logger.debug(f"Reading model file {name}.")
            self.set_geoms(gpd.read_file(fn, **kwargs), name=name)

    def write_geoms(self, fn: str = "geoms/{name}.geojson", **kwargs) -> None:
        """Write model geometries to a vector file (by default GeoJSON) at <root>/<fn>

        key-word arguments are passed to :py:meth:`geopandas.GeoDataFrame.to_file`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'geoms/{name}.geojson'
        """
        if len(self._geoms) == 0:
            self.logger.debug("No geoms data found, skip writing.")
            return
        self._assert_write_mode
        if "driver" not in kwargs:
            kwargs.update(driver="GeoJSON")  # default
        for name, gdf in self._geoms.items():
            if not isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)) or len(gdf) == 0:
                self.logger.warning(
                    f"{name} object of type {type(gdf).__name__} not recognized"
                )
                continue
            self.logger.debug(f"Writing file {fn.format(name=name)}")
            _fn = join(self.root, fn.format(name=name))
            if not isdir(dirname(_fn)):
                os.makedirs(dirname(_fn))
            gdf.to_file(_fn, **kwargs)

    # OLD model geometry files; TODO remove

    @property
    def staticgeoms(self):
        """This property will be deprecated in future versions, use :py:meth:`~hydromt.Model.geom`"""
        warnings.warn(
            "The staticgeoms method will be deprecated in future versions, use geoms instead.",
            DeprecationWarning,
        )
        if not self._geoms and self._read:
            self.read_staticgeoms()
        self._staticgeoms = self._geoms
        return self._staticgeoms

    def set_staticgeoms(self, geom: Union[gpd.GeoDataFrame, gpd.GeoSeries], name: str):
        """This method will be deprecated in future versions, use :py:meth:`~hydromt.Model.set_geoms`"""
        warnings.warn(
            "The set_staticgeoms method will be deprecated in future versions, use set_geoms instead.",
            DeprecationWarning,
        )
        return self.set_geoms(geom, name)

    def read_staticgeoms(self):
        """This method will be deprecated in future versions, use :py:meth:`~hydromt.Model.read_geoms`"""
        warnings.warn(
            'The read_staticgeoms" method will be deprecated in future versions, use read_geoms instead.',
            DeprecationWarning,
        )
        return self.read_geoms(fn="staticgeoms/*.geojson")

    def write_staticgeoms(self):
        """This method will be deprecated in future versions, use :py:meth:`~hydromt.Model.write_geoms`"""
        warnings.warn(
            'The "write_staticgeoms" method will be deprecated in future versions, use  "write_geoms" instead.',
            DeprecationWarning,
        )
        return self.write_geoms(fn="staticgeoms/{name}.geojson")

    # model forcing files
    @property
    def forcing(self) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """Model forcing. Returns dict of xarray.DataArray or xarray.Dataset"""
        if not self._forcing and self._read:
            self.read_forcing()
        return self._forcing

    def set_forcing(
        self,
        data: Union[xr.DataArray, xr.Dataset],
        name: Optional[str] = None,
        split_dataset: Optional[bool] = True,
    ):
        """Add data to forcing attribute.

        Arguments
        ---------
        data: xarray.Dataset or xarray.DataArray
            New forcing data to add
        name: str, optional
            Results name, required if data is xarray.Dataset is and split_dataset=False.
        split_dataset: bool, optional
            If True (default), split a Dataset to store each variable as a DataArray.
        """
        data_dict = _check_data(data, name, split_dataset)
        for name in data_dict:
            if name in self._forcing:
                self.logger.warning(f"Replacing forcing: {name}")
            self._forcing[name] = data_dict[name]

    def read_forcing(self, fn: str = "forcing/*.nc", **kwargs) -> None:
        """Read forcing at <root>/<fn> and add to forcing property

        key-word arguments are passed to :py:func:`xarray.open_dataset`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may wildcards, by default "forcing/*.nc"
        """
        self._assert_read_mode
        ncs = self._read_nc(fn, **kwargs)
        for name, ds in ncs.items():
            self.set_forcing(ds, name=name)

    def write_forcing(self, fn="forcing/{name}.nc", **kwargs) -> None:
        """Write forcing to netcdf file at <root>/<fn>

        key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'forcing/{name}.nc'
        """
        if len(self._forcing) == 0:
            self.logger.debug("No forcing data found, skip writing.")
        else:
            self._assert_write_mode
            self._write_nc(self._forcing, fn, **kwargs)

    # model state files
    @property
    def states(self) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """Model states. Returns dict of xarray.DataArray or xarray.Dataset"""
        if not self._states and self._read:
            self.read_states()
        return self._states

    def set_states(
        self,
        data: Union[xr.DataArray, xr.Dataset],
        name: Optional[str] = None,
        split_dataset: Optional[bool] = True,
    ):
        """Add data to states attribute.

        Arguments
        ---------
        data: xarray.Dataset or xarray.DataArray
            New forcing data to add
        name: str, optional
            Results name, required if data is xarray.Dataset and split_dataset=False.
        split_dataset: bool, optional
            If True (default), split a Dataset to store each variable as a DataArray.
        """
        data_dict = _check_data(data, name, split_dataset)
        for name in data_dict:
            if name in self._states:
                self.logger.warning(f"Replacing state: {name}")
            self._states[name] = data_dict[name]

    def read_states(self, fn: str = "states/*.nc", **kwargs) -> None:
        """Read states at <root>/<fn> and add to states property

        key-word arguments are passed to :py:func:`xarray.open_dataset`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may wildcards, by default "states/*.nc"
        """
        self._assert_read_mode
        ncs = self._read_nc(fn, **kwargs)
        for name, ds in ncs.items():
            self.set_states(ds, name=name)

    def write_states(self, fn="states/{name}.nc", **kwargs) -> None:
        """Write states to netcdf file at <root>/<fn>

        key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'states/{name}.nc'
        """
        if len(self._states) == 0:
            self.logger.debug("No states data found, skip writing.")
        else:
            self._assert_write_mode
            self._write_nc(self._states, fn, **kwargs)

    # model results files; NOTE we don't have a write_results method (that's up to the model kernel)
    @property
    def results(self) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """Model results.  Returns dict of xarray.DataArray or xarray.Dataset"""
        if not self._results and self._read:
            self.read_results()
        return self._results

    def set_results(
        self,
        data: Union[xr.DataArray, xr.Dataset],
        name: Optional[str] = None,
        split_dataset: Optional[bool] = False,
    ):
        """Add data to results attribute.

        Dataset can either be added as is (default) or split into several
        DataArrays using the split_dataset argument.

        Arguments
        ---------
        data: xarray.Dataset or xarray.DataArray
            New forcing data to add
        name: str, optional
            Results name, required if data is xarray.Dataset and split_dataset=False.
        split_dataset: bool, optional
            If True (False by default), split a Dataset to store each variable as a DataArray.
        """
        data_dict = _check_data(data, name, split_dataset)
        for name in data_dict:
            if name in self._results:
                self.logger.warning(f"Replacing result: {name}")
            self._results[name] = data_dict[name]

    def read_results(self, fn: str = "results/*.nc", **kwargs) -> None:
        """Read results at <root>/<fn> and add to results property

        key-word arguments are passed to :py:func:`xarray.open_dataset`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may wildcards, by default "results/*.nc"
        """
        self._assert_read_mode
        ncs = self._read_nc(fn, **kwargs)
        for name, ds in ncs.items():
            self.set_results(ds, name=name)

    def _write_nc(
        self, nc_dict: Dict[str, Union[xr.DataArray, xr.Dataset]], fn, **kwargs
    ) -> None:
        for name, ds in nc_dict.items():
            if not isinstance(ds, (xr.Dataset, xr.DataArray)) or len(ds) == 0:
                self.logger.error(
                    f"{name} object of type {type(ds).__name__} not recognized"
                )
                continue
            self.logger.debug(f"Writing file {fn.format(name=name)}")
            _fn = join(self.root, fn.format(name=name))
            if not isdir(dirname(_fn)):
                os.makedirs(dirname(_fn))
            ds.to_netcdf(_fn, **kwargs)

    # general reader & writer
    def _read_nc(
        self, fn: str, mask_and_scale=False, single_var_as_array=True, **kwargs
    ) -> Dict[str, xr.Dataset]:
        ncs = dict()
        fns = glob.glob(join(self.root, fn))
        if "chunks" not in kwargs:  # read lazy by default
            kwargs.update(chunks="auto")
        for fn in fns:
            name = basename(fn).split(".")[0]
            self.logger.debug(f"Reading model file {name}.")
            ds = xr.open_dataset(fn, mask_and_scale=mask_and_scale, **kwargs)
            # set geo coord if present as coordinate of dataset
            if GEO_MAP_COORD in ds.data_vars:
                ds = ds.set_coords(GEO_MAP_COORD)
            # single-variable Dataset to DataArray
            if single_var_as_array and len(ds.data_vars) == 1:
                (ds,) = ds.data_vars.values()
            ncs.update({name: ds})
        return ncs

    ## properties / methods below can be used directly in actual class
    @property
    def crs(self) -> CRS:
        """Returns coordinate reference system embedded in region."""
        if len(self._staticmaps) > 0:
            return self.staticmaps.raster.crs
        else:
            return self.region.crs

    def set_crs(self, crs) -> None:
        warnings.warn(
            '"set_crs" is deprecated. Please set the crs of all model components instead.',
            DeprecationWarning,
        )
        if len(self._staticmaps) > 0:
            return self.staticmaps.raster.set_crs(crs)

    @property
    def dims(self) -> Tuple:
        """Returns spatial dimension names of staticmaps.
        ..NOTE: will be deprecated in future versions"""
        if len(self._staticmaps) > 0:
            return self.staticmaps.raster.dims

    @property
    def coords(self) -> Dict:
        """Returns the coordinates of model staticmaps.
        ..NOTE: will be deprecated in future versions"""
        if len(self._staticmaps) > 0:
            return self.staticmaps.raster.coords

    @property
    def res(self) -> Tuple:
        """Returns the resolution of the model staticmaps.
        ..NOTE: will be deprecated in future versions"""
        if len(self._staticmaps) > 0:
            return self.staticmaps.raster.res

    @property
    def transform(self):
        """Returns the geospatial transform of the model staticmaps.
        ..NOTE: will be deprecated in future versions"""
        if len(self._staticmaps) > 0:
            return self.staticmaps.raster.transform

    @property
    def width(self):
        """Returns the width of the model staticmaps.
        ..NOTE: will be deprecated in future versions"""
        if len(self._staticmaps) > 0:
            return self.staticmaps.raster.width

    @property
    def height(self):
        """Returns the height of the model staticmaps.
        ..NOTE: will be deprecated in future versions"""
        if len(self._staticmaps) > 0:
            return self.staticmaps.raster.height

    @property
    def shape(self) -> Tuple:
        """Returns the shape of the model staticmaps.
        ..NOTE: will be deprecated in future versions"""
        if len(self._staticmaps) > 0:
            return self.staticmaps.raster.shape

    @property
    def bounds(self) -> Tuple:
        """Returns the bounding box of the model region."""
        if len(self._staticmaps) > 0:
            return self.staticmaps.raster.bounds
        else:
            return self.region.total_bounds

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns the geometry of the model area of interest."""
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        # TODO: For now stays here but move to grid in GridModel and delete
        elif len(self.staticmaps) > 0:
            warnings.warn(
                'Defining "region" based on staticmaps will be deprecated. Either use use region from GridModel or define your own method.',
                DeprecationWarning,
            )
            crs = self.staticmaps.raster.crs
            if crs is None and hasattr(crs, "to_epsg"):
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            region = gpd.GeoDataFrame(
                geometry=[box(*self.staticmaps.raster.bounds)], crs=crs
            )
        return region

    # test methods
    def test_model_api(self):
        warnings.warn(
            '"test_model_api" is now part of the internal API, use "_test_model_api" instead.',
            DeprecationWarning,
        )
        return self._test_model_api()

    def _test_model_api(self) -> List:
        """Test compliance with HydroMT Model API.

        Returns
        -------
        non_compliant: list
            List of model components that are non-compliant with the model API structure.
        """
        non_compliant = []
        for component, dtype in self.api.items():
            obj = getattr(self, component, None)
            try:
                assert obj is not None, component
                _assert_isinstance(obj, dtype, component)
            except AssertionError as err:
                non_compliant.append(str(err))

        return non_compliant

    def _test_equal(self, other, skip_component=["root"]) -> Tuple[bool, Dict]:
        """Test if two models including their data components are equal

        Parameters
        ----------
        other : Model (or subclass)
            Model to compare against
        skip_component: list
            List of components to skip when testing equality. By default root.

        Returns
        -------
        equal: bool
            True if equal
        errors: dict
            Dictionary with errors per model component which is not equal
        """
        assert isinstance(other, type(self))
        components = list(self.api.keys())
        components_other = list(other.api.keys())
        assert components == components_other
        for cp in skip_component:
            if cp in components:
                components.remove(cp)
        errors = {}
        for prop in components:
            errors.update(
                **_check_equal(getattr(self, prop), getattr(other, prop), prop)
            )
        return len(errors) == 0, errors


def _check_data(
    data: Union[xr.DataArray, xr.Dataset],
    name: Optional[str] = None,
    split_dataset=True,
) -> Dict:
    if isinstance(data, xr.DataArray):
        # NOTE name can be different from data.name !
        if data.name is None and name is not None:
            data.name = name
        elif name is None and data.name is not None:
            name = data.name
        elif data.name is None and name is None:
            raise ValueError("Name required for DataArray.")
        data = {name: data}
    elif isinstance(data, xr.Dataset):  # return dict for consistency
        if split_dataset:
            data = {name: data[name] for name in data.data_vars}
        elif name is None:
            raise ValueError("Name required for Dataset.")
        else:
            data = {name: data}
    else:
        raise ValueError(f'Data type "{type(data).__name__}" not recognized')
    return data


def _assert_isinstance(obj: Any, dtype: Any, name: str = ""):
    """Check if obj match typing or class (dtype)"""
    args = typing.get_args(dtype)
    _cls = typing.get_origin(dtype)
    if len(args) == 0 and dtype != Any and dtype is not None:
        assert isinstance(obj, dtype), name
    elif _cls == Union:
        assert isinstance(obj, args), name
    elif _cls is not None:
        assert isinstance(obj, _cls), name
    # recursive check of dtype dict keys and values
    if len(args) > 0 and _cls is dict:
        for key, val in obj.items():
            _assert_isinstance(key, args[0], f"{name}.{str(key)}")
            _assert_isinstance(val, args[1], f"{name}.{str(key)}")


def _check_equal(a, b, name="") -> Dict[str, str]:
    """Recursive test of model components.
    Returns dict with component name and associated error message."""
    errors = {}
    try:
        assert isinstance(b, type(a)), "property types do not match"
        if isinstance(a, dict):
            for key in a:
                assert key in b, f"{key} missing"
                errors.update(**_check_equal(a[key], b[key], f"{name}.{key}"))
        elif isinstance(a, (xr.DataArray, xr.Dataset)):
            xr.testing.assert_allclose(a, b)
        elif isinstance(a, gpd.GeoDataFrame):
            assert_geodataframe_equal(a, b, check_like=True, check_less_precise=True)
        elif isinstance(a, np.ndarray):
            np.testing.assert_allclose(a, b)
        else:
            assert a == b, "values not equal"
    except AssertionError as e:
        errors.update({name: e})
    return errors
