# -*- coding: utf-8 -*-
"""General and basic API for models in HydroMT."""

import glob
import logging
import os
import shutil
import typing
from abc import ABCMeta
from inspect import _empty, signature
from os.path import abspath, basename, dirname, isabs, isdir, isfile, join
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS

from hydromt import hydromt_step
from hydromt._typing import DeferedFileClose, StrPath, XArrayDict
from hydromt._utils import _classproperty
from hydromt._utils.rgetattr import rgetattr
from hydromt._utils.steps_validator import validate_steps
from hydromt.components import ModelRegionComponent
from hydromt.components.base import ModelComponent
from hydromt.data_catalog import DataCatalog
from hydromt.gis.raster import GEO_MAP_COORD
from hydromt.io import configread
from hydromt.io.writers import configwrite
from hydromt.plugins import PLUGINS
from hydromt.root import ModelRoot
from hydromt.utils.deep_merge import deep_merge

__all__ = ["Model"]

_logger = logging.getLogger(__name__)
T = TypeVar("T", bound=ModelComponent)


class Model(object, metaclass=ABCMeta):
    """
    General and basic API for models in HydroMT.

    Inherit from this class to pre-define mandatory components in the model.
    """

    _DATADIR = ""  # path to the model data folder
    _NAME: str = "modelname"
    _CONF: StrPath = "model.yml"
    _GEOMS = {"<general_hydromt_name>": "<model_name>"}
    _MAPS = {"<general_hydromt_name>": "<model_name>"}
    _FOLDERS = [""]
    _TMP_DATA_DIR = None
    # supported model version should be filled by the plugins
    # e.g. _MODEL_VERSION = ">=1.0, <1.1"
    _MODEL_VERSION = None

    _API = {
        "crs": CRS,
        "config": Dict[str, Any],
        "geoms": Dict[str, gpd.GeoDataFrame],
        "maps": XArrayDict,
        "forcing": XArrayDict,
        "region": ModelRegionComponent,
        "results": XArrayDict,
        "states": XArrayDict,
    }

    def __init__(
        self,
        components: Optional[dict[str, dict[str, Any]]] = None,
        root: Optional[str] = None,
        mode: str = "w",
        config_fn: Optional[str] = None,
        data_libs: Optional[Union[List, str]] = None,
        logger=_logger,
        **artifact_keys,
    ):
        r"""Initialize a model.

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
            List of data catalog configuration files, by default None
        \**artifact_keys:
            Additional keyword arguments to be passed down.
        logger:
            The logger to be used.
        """
        # Recursively update the options with any defaults that are missing in the configuration.
        components = components or {}
        components = deep_merge(
            {"region": {"type": "ModelRegionComponent"}}, components
        )

        data_libs = data_libs or []

        self.logger = logger

        # link to data
        self.data_catalog = DataCatalog(
            data_libs=data_libs, logger=self.logger, **artifact_keys
        )

        # placeholders
        # metadata maps that can be at different resolutions
        self._config: Optional[Dict[str, Any]] = None  # nested dictionary
        self._maps: Optional[XArrayDict] = None

        self._geoms: Optional[Dict[str, gpd.GeoDataFrame]] = None
        self._forcing: Optional[XArrayDict] = None
        self._states: Optional[XArrayDict] = None
        self._results: Optional[XArrayDict] = None

        # file system
        self.root: ModelRoot = ModelRoot(root or ".", mode=mode)

        self._components: Dict[str, ModelComponent] = {}
        self._add_components(components)

        self._defered_file_closes = []

        # model paths
        self._config_fn = self._CONF if config_fn is None else config_fn

        model_metadata = cast(
            Dict[str, str], PLUGINS.model_metadata[self.__class__.__name__]
        )
        self.logger.info(
            f"Initializing {self._NAME} model from {model_metadata['plugin_name']} (v{model_metadata['version']})."
        )

    def _add_components(self, components: dict[str, dict[str, Any]]) -> None:
        """Add all components that are specified in the config file."""
        for name, options in components.items():
            type_name = options.pop("type")
            component_type = PLUGINS.component_plugins[type_name]
            self.add_component(name, component_type(self, **options))

    def add_component(self, name: str, component: ModelComponent) -> None:
        """Add a component to the model. Will raise an error if the component already exists."""
        if name in self._components:
            raise ValueError(f"Component {name} already exists in the model.")
        if not name.isidentifier():
            raise ValueError(f"Component name {name} is not a valid identifier.")
        self._components[name] = component

    def get_component(self, name: str, _: Type[T]) -> T:
        """Get a component from the model. Will raise an error if the component does not exist."""
        return cast(T, self._components[name])

    def __getattr__(self, name: str) -> ModelComponent:
        """Get a component from the model. Will raise an error if the component does not exist."""
        return self._components[name]

    @property
    def region(self) -> ModelRegionComponent:
        """Return the model region component."""
        return self.get_component("region", ModelRegionComponent)

    @_classproperty
    def api(cls) -> Dict:
        """Return all model components and their data types."""
        _api = cls._API.copy()

        # reversed is so that child attributes take priority
        # this does mean that it becomes imporant in which order you
        # inherit from your base classes.
        for base_cls in reversed(cls.__mro__):
            if hasattr(base_cls, "_API"):
                _api.update(getattr(base_cls, "_API", {}))
        return _api

    def build(
        self,
        *,
        region: dict[str, Any],
        write: Optional[bool] = True,
        steps: Optional[list[dict[str, dict[str, Any]]]] = None,
    ):
        r"""Single method to build a model from scratch based on settings in `steps`.

        Methods will be run one by one based on the /order of appearance in `steps`
        (configuration file). For a list of available functions see :ref:`The model API<model_api>`
        and :ref:`The plugin documentation<plugin_create>`

        By default the full model will be written at the end, except if a write step
        is called for somewhere in steps, then this is skipped.

        Note that the \* in the signature signifies that all of the arguments to this function
        MUST be provided as keyword arguments.

        Parameters
        ----------
        region: dict
            Description of model region. See :py:meth:`~hydromt.workflows.parse_region`
            for all options.
        write: bool, optional
            Write complete model after executing all methods in opt, by default True.
        steps: Optional[list[dict[str, dict[str, Any]]]]
            Model build configuration. The configuration can be parsed from a
            configuration file using :py:meth:`~hydromt.io.readers.configread`.
            This is a list of nested dictionary where the first-level keys are the names
            of the method for a ``Model`` method (e.g. `write`) OR the name of a component followed by the name of the method to run separated by a dot for ``ModelComponent`` method (e.g. `grid.write`).
            Any subsequent pairs will be passed to the method as arguments.

            .. code-block:: text

                [
                    - <component_name>.<name of method1>: {
                        <argument1>: <value1>, <argument2>: <value2>
                    },
                    - <component_name>.<name of method2>: {
                        ...
                    }
                ]

        """
        steps = steps or []
        validate_steps(self, steps)
        self._update_region_from_arguments(steps, region)
        self._move_region_create_to_front(steps)

        for step_dict in steps:
            step, kwargs = next(iter(step_dict.items()))
            self.logger.info(f"build: {step}")
            # Call the methods.
            method = rgetattr(self, step)
            params = {
                param: arg.default
                for param, arg in signature(method).parameters.items()
                if arg.default != _empty
            }
            merged = {**params, **kwargs}
            for k, v in merged.items():
                self.logger.info(f"{method}.{k}: {v}")
            method(**kwargs)

        # If there are any write options included in the steps,
        # we don't need to write the whole model.
        if write and not self._options_contain_write(steps):
            self.write()

    def update(
        self,
        *,
        model_out: Optional[StrPath] = None,
        write: Optional[bool] = True,
        steps: Optional[list[dict[str, dict[str, Any]]]] = None,
        forceful_overwrite: bool = False,
    ):
        r"""Single method to update a model based the settings in `steps`.

        Methods will be run one by one based on the /order of appearance in `steps`
        (configuration file). For a list of available functions see :ref:`The model API<model_api>`
        and :ref:`The plugin documentation<plugin_create>`

        Note that the \* in the signature signifies that all of the arguments to this function
        MUST be provided as keyword arguments.


        Parameters
        ----------
        model_out: str, path, optional
            Destination folder to write the model schematization after updating
            the model. If None the updated model components are overwritten in the
            current model schematization if these exist. By default None.
        write: bool, optional
            Write the updated model schematization to disk. By default True.
        steps: Optional[list[dict[str, dict[str, Any]]]]
            Model build configuration. The configuration can be parsed from a
            configuration file using :py:meth:`~hydromt.io.readers.configread`.
            This is a list of nested dictionary where the first-level keys are the names
            of a component followed by the name of the method to run seperated by a dot.
            anny subsequent pairs will be passed to the method as arguments.

            .. code-block:: text

                [
                    - <component_name>.<name of method1>: {
                        <argument1>: <value1>, <argument2>: <value2>
                    },
                    - <component_name>.<name of method2>: {
                        ...
                    }
                ]
          forceful_overwrite:
            Force open files to close when attempting to write them. In the case you
            try to write to a file that's already opened. The output will be written
            to a temporary file in case the original file cannot be written to.
        """
        steps = steps or []
        validate_steps(self, steps)

        # check if region.create is in the steps, and remove it.
        self._remove_region_create(steps)

        # read current model
        if not self.root.is_writing_mode():
            if model_out is None:
                raise ValueError(
                    '"model_out" directory required when updating in "read-only" mode.'
                )
            self.read()
            mode = "w+" if forceful_overwrite else "w"
            self.root.set(model_out, mode=mode)

        # check if model has a region
        if self.region.data is None:
            raise ValueError("Model region not found, setup model using `build` first.")

        # loop over methods from config file
        for step_dict in steps:
            step, kwargs = next(iter(step_dict.items()))
            self.logger.info(f"update: {step}")
            # Call the methods.
            method = rgetattr(self, step)
            params = {
                param: arg.default
                for param, arg in signature(method).parameters.items()
                if arg.default != _empty
            }
            merged = {**params, **kwargs}
            for k, v in merged.items():
                self.logger.info(f"{method}.{k}: {v}")
            method(**kwargs)

        # If there are any write options included in the steps,
        # we don't need to write the whole model.
        if write and not self._options_contain_write(steps):
            self.write()

        self._cleanup(forceful_overwrite=forceful_overwrite)

    @hydromt_step
    def write(self, components: Optional[List[str]] = None) -> None:
        """Write provided components to disk with defaults.

        Parameters
        ----------
            components: Optional[List[str]]
                the components that should be writen to disk. If None is provided
                all components will be written.
        """
        components = components or list(self._components.keys())
        for c in [self._components[name] for name in components]:
            c.write()

    @hydromt_step
    def read(self, components: Optional[List[str]] = None) -> None:
        """Read provided components from disk.

        Parameters
        ----------
            components: Optional[List[str]]
                the components that should be read from disk. If None is provided
                all components will be read.
        """
        self.logger.info(f"Reading model data from {self.root.path}")
        components = components or list(self._components.keys())
        for c in [self._components[name] for name in components]:
            c.read()

    @staticmethod
    def _options_contain_write(steps: list[dict[str, dict[str, Any]]]) -> bool:
        return any(
            next(iter(step_dict)).split(".")[-1] == "write" for step_dict in steps
        )

    @staticmethod
    def _update_region_from_arguments(
        steps: list[dict[str, dict[str, Any]]], region: dict[str, Any]
    ) -> None:
        try:
            region_step = next(
                step_dict
                for step_dict in enumerate(steps)
                if next(iter(step_dict)) == "region.create"
            )
            region_step[1]["region"] = region
        except StopIteration:
            steps.insert(0, {"region.create": {"region": region}})

    @staticmethod
    def _move_region_create_to_front(steps: list[dict[str, dict[str, Any]]]) -> None:
        region_create = next(
            step_dict for step_dict in steps if "region.create" in step_dict
        )
        steps.remove(region_create)
        steps.insert(0, region_create)

    def _remove_region_create(self, steps: list[dict[str, dict[str, Any]]]) -> None:
        try:
            steps.remove(
                next(step_dict for step_dict in steps if "region.create" in step_dict)
            )
            self.logger.warning(
                "region.create can only be called when building a model."
            )
        except StopIteration:
            pass

    @hydromt_step
    def write_data_catalog(
        self,
        root: Optional[StrPath] = None,
        data_lib_fn: StrPath = "hydromt_data.yml",
        used_only: bool = True,
        append: bool = True,
        save_csv: bool = False,
    ):
        """Write the data catalog to data_lib_fn.

        Parameters
        ----------
        root: str, Path, optional
            Global root for all relative paths in configuration file.
            If "auto" the data source paths are relative to the yaml output ``path``.
        data_lib_fn: str, Path, optional
            Path of output yml file, absolute or relative to the model root,
            by default "hydromt_data.yml".
        used_only: bool, optional
            If True, export only data entries kept in used_data list. By default True
        append: bool, optional
            If True, append to an existing
        save_csv: bool, optional
            If True, save the data catalog also as an csv table. By default False.
        """
        self.root._assert_write_mode()
        path = data_lib_fn if isabs(data_lib_fn) else join(self.root.path, data_lib_fn)
        cat = DataCatalog(logger=self.logger, fallback_lib=None)
        # read hydromt_data yml file and add to data catalog
        if self.root.is_reading_mode() and isfile(path) and append:
            cat.from_yml(path)
        # update data catalog with new used sources
        for name, source in self.data_catalog.iter_sources(used_only=used_only):
            cat.add_source(name, source)
        # write data catalog
        if cat.sources:
            if save_csv:
                csv_path = os.path.splitext(path)[0] + ".csv"
                cat.to_dataframe().reset_index().to_csv(
                    csv_path, sep=",", index=False, header=True
                )

            cat.to_yml(path, root=root)

    def test_equal(self, other: "Model") -> tuple[bool, dict[str, str]]:
        """Test if two models are equal, based on their components.

        Parameters
        ----------
        other : Model
            The model to compare against.

        Returns
        -------
        tuple[bool, dict[str, str]]
            True if equal, dictionary with errors per model component which is not equal.
        """
        assert isinstance(other, self.__class__)
        components = list(self._components.keys())
        components_other = list(other._components.keys())
        assert components == components_other
        errors: dict[str, str] = {}
        is_equal = True
        for name, c in self._components.items():
            component_equal, component_errors = c.test_equal(other._components[name])
            is_equal &= component_equal
            errors.update(**component_errors)
        return is_equal, errors

    # model configuration
    @property
    def config(self) -> Dict[str, Union[Dict, str]]:
        """Model configuration. Returns a (nested) dictionary."""
        if self._config is None:
            self._initialize_config()
        return self._config

    def _initialize_config(self, skip_read=False) -> None:
        """Initialize the model config."""
        if self._config is None:
            self._config = dict()
            if not skip_read:
                # no check for read mode here
                # model config is read if in read-mode and it exists
                # default config if in write-mode
                self.read_config()

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
        self._initialize_config()
        if len(args) < 2:
            raise TypeError("set_config() requires a least one key and one value.")
        args = list(args)
        value = args.pop(-1)
        if len(args) == 1 and "." in args[0]:
            args = args[0].split(".") + args[1:]
        branch = self._config
        for key in args[:-1]:
            if key not in branch or not isinstance(branch[key], dict):
                branch[key] = {}
            branch = branch[key]
        branch[args[-1]] = value

    def read_config(self, config_fn: Optional[str] = None):
        """Parse config from file.

        If no config file found a default config file is returned in writing mode.
        """
        self.root._assert_write_mode()
        prefix = "User defined"
        if config_fn is None:  # prioritize user defined config path (new v0.4.1)
            if not self.root.is_reading_mode():  # write-only mode > read default config
                config_fn = join(self._DATADIR, self._NAME, self._CONF)
                prefix = "Default"
            elif self.root is not None:  # append or write mode > read model config
                config_fn = join(self.root.path, self._config_fn)
                prefix = "Model"
        cfdict = dict()
        if config_fn is not None:
            if isfile(config_fn):
                cfdict = self._configread(config_fn)
                self.logger.debug(f"{prefix} config read from {config_fn}")
            elif (
                self.root is not None
                and not isabs(config_fn)
                and isfile(join(self.root.path, config_fn))
            ):
                cfdict = self._configread(join(self.root.path, config_fn))
                self.logger.debug(
                    f"{prefix} config read from {join(self.root.path,config_fn)}"
                )
            elif isfile(abspath(config_fn)):
                cfdict = self._configread(abspath(config_fn))
                self.logger.debug(f"{prefix} config read from {abspath(config_fn)}")
            else:  # skip for missing default
                self.logger.error(f"{prefix} config file not found at {config_fn}")

        # always overwrite config when reading
        self._config = cfdict

    def write_config(
        self, config_name: Optional[str] = None, config_root: Optional[str] = None
    ):
        """Write config to <root/config_fn>."""
        self.root._assert_write_mode()
        if config_name is not None:
            self._config_fn = config_name
        elif self._config_fn is None:
            self._config_fn = self._CONF
        if config_root is None:
            config_root = self.root.path
        fn = join(config_root, self._config_fn)
        self.logger.info(f"Writing model config to {fn}")
        self._configwrite(fn)

    def _configread(self, fn: str):
        return configread(fn, abs_path=False)

    def _configwrite(self, fn: str):
        return configwrite(fn, self.config)

    def setup_config(self, **cfdict):
        """Update config with a dictionary."""
        # TODO rename to update_config
        if len(cfdict) > 0:
            self.logger.debug("Setting model config options.")
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
            If True return the absolute path relative to the model root,
            by deafult False.
            NOTE: this assumes the config is located in model root!

        Returns
        -------
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
                value = Path(abspath(join(self.root.path, value)))
        return value

    # map files setup methods
    def setup_maps_from_rasterdataset(
        self,
        raster_fn: Union[str, Path, xr.Dataset],
        variables: Optional[List] = None,
        fill_method: Optional[str] = None,
        name: Optional[str] = None,
        reproject_method: Optional[str] = None,
        split_dataset: Optional[bool] = True,
        rename: Optional[Dict] = None,
    ) -> List[str]:
        """HYDROMT CORE METHOD: Add data variable(s) from ``raster_fn`` to maps object.

        If raster is a dataset, all variables will be added unless ``variables``
        list is specified.

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
            Name of new dataset in self.maps dictionnary,
            only in case split_dataset=False.
        reproject_method: str, optional
            See rasterio.warp.reproject for existing methods, by default the data is
            not reprojected (None).
        split_dataset: bool, optional
            If data is a xarray.Dataset split it into several xarray.DataArrays.
        rename: dict, optional
            Dictionary to rename variable names in raster_fn before adding to maps
            {'name_in_raster_fn': 'name_in_maps'}. By default empty.

        Returns
        -------
        list
            Names of added model map layers
        """
        rename = rename or {}
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
        # Rename and add to maps
        self.set_maps(ds.rename(rename), name=name, split_dataset=split_dataset)

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
        rename: Optional[Dict] = None,
        **kwargs,
    ) -> List[str]:
        r"""HYDROMT CORE METHOD: Add data variable(s) to maps object by reclassifying the data in ``raster_fn`` based on ``reclass_table_fn``.

        This is done by reclassifying the data in
        ``raster_fn`` based on ``reclass_table_fn``.

        Adds model layers:

        * **reclass_variables** maps: reclassified raster data

        Parameters
        ----------
        raster_fn: str, Path, xr.DataArray
            Data catalog key, path to raster file or raster xarray data object.
            Should be a DataArray. Else use `variable` argument for selection.
        reclass_table_fn: str, Path, pd.DataFrame
            Data catalog key, path to tabular data file or tabular pandas dataframe
            object for the reclassification table of `raster_fn`.
        reclass_variables: list
            List of reclass_variables from reclass_table_fn table to add to maps. Index
            column should match values in `raster_fn`.
        variable: str, optional
            Name of raster dataset variable to use. This is only required when reading
            datasets with multiple variables. By default None.
        fill_method : str, optional
            If specified, fills nodata values in `raster_fn` using fill_nodata method
            before reclassifying. Available methods are {'linear', 'nearest',
            'cubic', 'rio_idw'}.
        reproject_method: str, optional
            See rasterio.warp.reproject for existing methods, by default the data is
            not reprojected (None).
        name: str, optional
            Name of new maps variable, only in case split_dataset=False.
        split_dataset: bool, optional
            If data is a xarray.Dataset split it into several xarray.DataArrays.
        rename: dict, optional
            Dictionary to rename variable names in reclass_variables before adding to
            grid {'name_in_reclass_table': 'name_in_grid'}. By default empty.
        \**kwargs:
            Additional keyword arguments that are passed to the
            `data_catalog.get_rasterdataset` function.

        Returns
        -------
        list
            Names of added model map layers
        """  # noqa: E501
        rename = rename or {}
        self.logger.info(
            f"Preparing map data by reclassifying the data in {raster_fn} based"
            f" on {reclass_table_fn}"
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
        self.set_maps(ds_vars.rename(rename), name=name, split_dataset=split_dataset)

        return list(ds_vars.data_vars.keys())

    # model map
    @property
    def maps(self) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """Model maps. Returns dict of xarray.DataArray or xarray.Dataset."""
        if self._maps is None:
            self._initialize_maps()
        return self._maps

    def _initialize_maps(self, skip_read=False) -> None:
        """Initialize maps."""
        if self._maps is None:
            self._maps = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read_maps()

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
            Variable name, only in case data is of type DataArray or if a Dataset is
            added as is (split_dataset=False).
        split_dataset: bool, optional
            If data is a xarray.Dataset split it into several xarray.DataArrays.
        """
        self._initialize_maps()
        data_dict = _check_data(data, name, split_dataset)
        for name in data_dict:
            if name in self._maps:
                self.logger.warning(f"Replacing result: {name}")
            self._maps[name] = data_dict[name]

    def read_maps(self, fn: str = "maps/*.nc", **kwargs) -> None:
        r"""Read model map at <root>/<fn> and add to maps component.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.read_nc`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may contain wildcards,
            by default ``maps/\*.nc``
        kwargs:
            Additional keyword arguments that are passed to the
            `read_nc` function.
        """
        self.root._assert_read_mode()
        self._initialize_maps(skip_read=True)
        ncs = self.read_nc(fn, **kwargs)
        for name, ds in ncs.items():
            self.set_maps(ds, name=name)

    def write_maps(self, fn="maps/{name}.nc", **kwargs) -> None:
        r"""Write maps to netcdf file at <root>/<fn>.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.write_nc`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'maps/{name}.nc'
        \**kwargs:
            Additional keyword arguments that are passed to the
            `write_nc` function.
        """
        self.root._assert_write_mode()
        if len(self.maps) == 0:
            self.logger.debug("No maps data found, skip writing.")
        else:
            self.write_nc(self.maps, fn, **kwargs)

    # model geometry files
    @property
    def geoms(self) -> Dict[str, Union[gpd.GeoDataFrame, gpd.GeoSeries]]:
        """Model geometries.

        Return dict of geopandas.GeoDataFrame or geopandas.GeoDataSeries
        ..NOTE: previously call staticgeoms.
        """
        if self._geoms is None:
            self._initialize_geoms()
        return self._geoms

    def _initialize_geoms(self, skip_read=False) -> None:
        """Initialize geoms."""
        if self._geoms is None:
            self._geoms = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read_geoms()

    def set_geoms(self, geom: Union[gpd.GeoDataFrame, gpd.GeoSeries], name: str):
        """Add data to the geoms attribute.

        Arguments
        ---------
        geom: geopandas.GeoDataFrame or geopandas.GeoSeries
            New geometry data to add
        name: str
            Geometry name.
        """
        self._initialize_geoms()
        gtypes = [gpd.GeoDataFrame, gpd.GeoSeries]
        if not np.any([isinstance(geom, t) for t in gtypes]):
            raise ValueError(
                "First parameter map(s) should be geopandas.GeoDataFrame"
                " or geopandas.GeoSeries"
            )
        if name in self._geoms:
            self.logger.warning(f"Replacing geom: {name}")
        if hasattr(self, "crs"):
            # Verify if a geom is set to model crs and if not sets geom to model crs
            if self.crs and self.crs != geom.crs:
                geom.to_crs(self.crs.to_epsg(), inplace=True)
        self._geoms[name] = geom

    def read_geoms(self, fn: str = "geoms/*.geojson", **kwargs) -> None:
        r"""Read model geometries files at <root>/<fn> and add to geoms property.

        key-word arguments are passed to :py:func:`geopandas.read_file`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may contain wildcards,
            by default ``geoms/\*.nc``
        **kwargs:
            Additional keyword arguments that are passed to the
            `geopandas.read_file` function.
        """
        self.root._assert_read_mode()
        self._initialize_geoms(skip_read=True)
        fns = glob.glob(join(self.root.path, fn))
        for fn in fns:
            name = basename(fn).split(".")[0]
            self.logger.debug(f"Reading model file {name}.")
            self.set_geoms(gpd.read_file(fn, **kwargs), name=name)

    def write_geoms(
        self, fn: str = "geoms/{name}.geojson", to_wgs84: bool = False, **kwargs
    ) -> None:
        r"""Write model geometries to a vector file (by default GeoJSON) at <root>/<fn>.

        key-word arguments are passed to :py:meth:`geopandas.GeoDataFrame.to_file`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'geoms/{name}.geojson'
        to_wgs84: bool, optional
            Option to enforce writing GeoJSONs with WGS84(EPSG:4326) coordinates.
        \**kwargs:
            Additional keyword arguments that are passed to the
            `geopandas.to_file` function.
        """
        self.root._assert_write_mode()
        if len(self.geoms) == 0:
            self.logger.debug("No geoms data found, skip writing.")
            return
        for name, gdf in self.geoms.items():
            if not isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)) or len(gdf) == 0:
                self.logger.warning(
                    f"{name} object of type {type(gdf).__name__} not recognized"
                )
                continue
            self.logger.debug(f"Writing file {fn.format(name=name)}")
            _fn = join(self.root.path, fn.format(name=name))
            if not isdir(dirname(_fn)):
                os.makedirs(dirname(_fn))
            if to_wgs84 and (
                kwargs.get("driver") == "GeoJSON"
                or str(fn).lower().endswith(".geojson")
            ):
                gdf = gdf.to_crs(4326)
            gdf.to_file(_fn, **kwargs)

    # model forcing files
    @property
    def forcing(self) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """Model forcing. Returns dict of xarray.DataArray or xarray.Dataset."""
        if self._forcing is None:
            self._initialize_forcing()
        return self._forcing

    def _initialize_forcing(self, skip_read=False) -> None:
        """Initialize forcing."""
        if self._forcing is None:
            self._forcing = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read_forcing()

    def set_forcing(
        self,
        data: Union[xr.DataArray, xr.Dataset, pd.DataFrame],
        name: Optional[str] = None,
        split_dataset: Optional[bool] = True,
    ):
        """Add data to forcing attribute.

        Data can be xarray.DataArray, xarray.Dataset or pandas.DataFrame.
        If pandas.DataFrame, indices should be the DataFrame index and the columns
        the variable names. the DataFrame will then be converted to xr.Dataset using
        :py:meth:`pandas.DataFrame.to_xarray` method.

        Arguments
        ---------
        data: xarray.Dataset or xarray.DataArray or pd.DataFrame
            New forcing data to add
        name: str, optional
            Results name, required if data is xarray.Dataset is and split_dataset=False.
        split_dataset: bool, optional
            If True (default), split a Dataset to store each variable as a DataArray.
        """
        self._initialize_forcing()
        if isinstance(data, pd.DataFrame):
            data = data.to_xarray()
        data_dict = _check_data(data, name, split_dataset)
        for name in data_dict:
            if name in self._forcing:
                self.logger.warning(f"Replacing forcing: {name}")
            self._forcing[name] = data_dict[name]

    def read_forcing(self, fn: str = "forcing/*.nc", **kwargs) -> None:
        """Read forcing at <root>/<fn> and add to forcing property.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.read_nc`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may contain wildcards,
            by default forcing/.nc
        kwargs:
            Additional keyword arguments that are passed to the `read_nc`
            function.
        """
        self._initialize_forcing(skip_read=True)
        ncs = self.read_nc(fn, **kwargs)
        for name, ds in ncs.items():
            self.set_forcing(ds, name=name)

    def write_forcing(self, fn="forcing/{name}.nc", **kwargs) -> None:
        """Write forcing to netcdf file at <root>/<fn>.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.write_nc`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'forcing/{name}.nc'
        kwargs:
            Additional keyword arguments that are passed to the `write_nc`
            function.
        """
        self.root._assert_read_mode()
        if len(self.forcing) == 0:
            self.logger.debug("No forcing data found, skip writing.")
        else:
            self.write_nc(self.forcing, fn, **kwargs)

    # model state files
    @property
    def states(self) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """Model states. Returns dict of xarray.DataArray or xarray.Dataset."""
        if self._states is None:
            self._initialize_states()
        return self._states

    def _initialize_states(self, skip_read=False) -> None:
        """Initialize states."""
        if self._states is None:
            self._states = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read_states()

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
        self._initialize_states()
        data_dict = _check_data(data, name, split_dataset)
        for name in data_dict:
            if name in self._states:
                self.logger.warning(f"Replacing state: {name}")
            self._states[name] = data_dict[name]

    def read_states(self, fn: str = "states/*.nc", **kwargs) -> None:
        r"""Read states at <root>/<fn> and add to states property.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.read_nc`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may contain wildcards,
            by default states/\*.nc
        kwargs:
            Additional keyword arguments that are passed to the `read_nc`
            function.
        """
        self.root._assert_read_mode()
        self._initialize_states(skip_read=True)
        ncs = self.read_nc(fn, **kwargs)
        for name, ds in ncs.items():
            self.set_states(ds, name=name, split_dataset=True)

    def write_states(self, fn="states/{name}.nc", **kwargs) -> None:
        """Write states to netcdf file at <root>/<fn>.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.write_nc`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'states/{name}.nc'
        **kwargs:
            Additional keyword arguments that are passed to the `write_nc`
            function.
        """
        self.root._assert_write_mode()
        if len(self.states) == 0:
            self.logger.debug("No states data found, skip writing.")
        else:
            self.write_nc(self.states, fn, **kwargs)

    # model results files; NOTE we don't have a write_results method
    # (that's up to the model kernel)
    @property
    def results(self) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """Model results. Returns dict of xarray.DataArray or xarray.Dataset."""
        if self._results is None:
            self._initialize_results()
        return self._results

    def _initialize_results(self, skip_read=False) -> None:
        """Initialize results."""
        if self._results is None:
            self._results = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read_results()

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
            If True (False by default), split a Dataset to store each variable
            as a DataArray.
        """
        self._initialize_results()
        data_dict = _check_data(data, name, split_dataset)
        for name in data_dict:
            if name in self._results:
                self.logger.warning(f"Replacing result: {name}")
            self._results[name] = data_dict[name]

    def read_results(self, fn: str = "results/*.nc", **kwargs) -> None:
        """Read results at <root>/<fn> and add to results property.

        key-word arguments are passed to :py:meth:`~hydromt.models.Model.read_nc`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may contain wildcards,
            by default ``results/*.nc``
        kwargs:
            Additional keyword arguments that are passed to the `read_nc`
            function.
        """
        self.root._assert_read_mode()
        self._initialize_results(skip_read=True)
        ncs = self.read_nc(fn, **kwargs)
        for name, ds in ncs.items():
            self.set_results(ds, name=name)

    # general reader & writer
    def _cleanup(self, forceful_overwrite=False, max_close_attempts=2) -> List[str]:
        """Try to close all defered file handles.

        Try to overwrite the destination file with the temporary one until either the
        maximum number of tries is reached or until it succeeds. The forced cleanup
        also attempts to close the original file handle, which could cause trouble
        if the user will try to read from the same file handle after this function
        is called.

        Parameters
        ----------
        forceful_overwrite: bool
            Attempt to force closing defered file handles before writing to them.
        max_close_attempts: int
            Number of times to try and overwrite the original file, before giving up.

        """
        failed_closes = []
        while len(self._defered_file_closes) > 0:
            close_handle = self._defered_file_closes.pop()
            if close_handle["close_attempts"] > max_close_attempts:
                # already tried to close this to many times so give up
                self.logger.error(
                    f"Max write attempts to file {close_handle['org_fn']}"
                    " exceeded. Skipping..."
                    f"Instead data was written to tmpfile: {close_handle['tmp_fn']}"
                )
                continue

            if forceful_overwrite:
                close_handle["ds"].close()
            try:
                shutil.move(close_handle["tmp_fn"], close_handle["org_fn"])
            except PermissionError:
                self.logger.error(
                    f"Could not write to destination file {close_handle['org_fn']} "
                    "because the following error was raised: {e}"
                )
                close_handle["close_attempts"] += 1
                self._defered_file_closes.append(close_handle)
                failed_closes.append((close_handle["org_fn"], close_handle["tmp_fn"]))

        return list(set(failed_closes))

    def write_nc(
        self,
        nc_dict: XArrayDict,
        fn: str,
        gdal_compliant: bool = False,
        rename_dims: bool = False,
        force_sn: bool = False,
        **kwargs,
    ) -> None:
        """Write dictionnary of xarray.Dataset and/or xarray.DataArray to netcdf files.

        Possibility to update the xarray objects attributes to get GDAL compliant NetCDF
        files, using :py:meth:`~hydromt.raster.gdal_compliant`.
        The function will first try to directly write to file. In case of
        PermissionError, it will first write a temporary file and add to the
        self._defered_file_closes attribute. Renaming and closing of netcdf filehandles
        will be done by calling the self._cleanup function.

        key-word arguments are passed to :py:meth:`xarray.Dataset.to_netcdf`

        Parameters
        ----------
        nc_dict: dict
            Dictionary of xarray.Dataset and/or xarray.DataArray to write
        fn: str
            filename relative to model root and should contain a {name} placeholder
        gdal_compliant: bool, optional
            If True, convert xarray.Dataset and/or xarray.DataArray to gdal compliant
            format using :py:meth:`~hydromt.raster.gdal_compliant`
        rename_dims: bool, optional
            If True, rename x_dim and y_dim to standard names depending on the CRS
            (x/y for projected and lat/lon for geographic). Only used if
            ``gdal_compliant`` is set to True. By default, False.
        force_sn: bool, optional
            If True, forces the dataset to have South -> North orientation. Only used
            if ``gdal_compliant`` is set to True. By default, False.
        **kwargs:
            Additional keyword arguments that are passed to the `to_netcdf`
            function.
        """
        for name, ds in nc_dict.items():
            if not isinstance(ds, (xr.Dataset, xr.DataArray)) or len(ds) == 0:
                self.logger.error(
                    f"{name} object of type {type(ds).__name__} not recognized"
                )
                continue
            self.logger.debug(f"Writing file {fn.format(name=name)}")
            _fn = join(self.root.path, fn.format(name=name))
            if not isdir(dirname(_fn)):
                os.makedirs(dirname(_fn))
            if gdal_compliant:
                ds = ds.raster.gdal_compliant(
                    rename_dims=rename_dims, force_sn=force_sn
                )
            try:
                ds.to_netcdf(_fn, **kwargs)
            except PermissionError:
                _logger.warning(f"Could not write to file {_fn}, defering write")
                if self._TMP_DATA_DIR is None:
                    self._TMP_DATA_DIR = TemporaryDirectory()

                tmp_fn = join(str(self._TMP_DATA_DIR), f"{_fn}.tmp")
                ds.to_netcdf(tmp_fn, **kwargs)
                self._defered_file_closes.append(
                    DeferedFileClose(
                        ds=ds,
                        org_fn=join(str(self._TMP_DATA_DIR), _fn),
                        tmp_fn=tmp_fn,
                        close_attempts=1,
                    )
                )

    def read_nc(
        self,
        fn: StrPath,
        mask_and_scale: bool = False,
        single_var_as_array: bool = True,
        load: bool = False,
        **kwargs,
    ) -> Dict[str, xr.Dataset]:
        """Read netcdf files at <root>/<fn> and return as dict of xarray.Dataset.

        NOTE: Unless `single_var_as_array` is set to False a single-variable data source
        will be returned as :py:class:`xarray.DataArray` rather than
        :py:class:`xarray.Dataset`.
        key-word arguments are passed to :py:func:`xarray.open_dataset`.

        Parameters
        ----------
        fn : str
            filename relative to model root, may contain wildcards
        mask_and_scale : bool, optional
            If True, replace array values equal to _FillValue with NA and scale values
            according to the formula original_values * scale_factor + add_offset, where
            _FillValue, scale_factor and add_offset are taken from variable attributes
            (if they exist).
        single_var_as_array : bool, optional
            If True, return a DataArray if the dataset consists of a single variable.
            If False, always return a Dataset. By default True.
        load : bool, optional
            If True, the data is loaded into memory. By default False.
        **kwargs:
            Additional keyword arguments that are passed to the `xr.open_dataset`
            function.

        Returns
        -------
        Dict[str, xr.Dataset]
            dict of xarray.Dataset
        """
        ncs = dict()
        fns = glob.glob(join(self.root.path, fn))
        if "chunks" not in kwargs:  # read lazy by default
            kwargs.update(chunks="auto")
        for fn in fns:
            name = basename(fn).split(".")[0]
            self.logger.debug(f"Reading model file {name}.")
            # Load data to allow overwritting in r+ mode
            if load:
                ds = xr.open_dataset(fn, mask_and_scale=mask_and_scale, **kwargs).load()
                ds.close()
            else:
                ds = xr.open_dataset(fn, mask_and_scale=mask_and_scale, **kwargs)
            # set geo coord if present as coordinate of dataset
            if GEO_MAP_COORD in ds.data_vars:
                ds = ds.set_coords(GEO_MAP_COORD)
            # single-variable Dataset to DataArray
            if single_var_as_array and len(ds.data_vars) == 1:
                (ds,) = ds.data_vars.values()
            ncs.update({name: ds})
        return ncs

    @property
    def crs(self) -> CRS:
        """Returns coordinate reference system embedded in region."""
        return self.region.crs


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
    """Check if obj match typing or class (dtype)."""
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
