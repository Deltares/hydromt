# -*- coding: utf-8 -*-
"""General and basic API for models in HydroMT."""

import glob
import logging
import os
import shutil
import typing
from abc import ABCMeta
from inspect import _empty, signature
from os.path import basename, dirname, isabs, isdir, isfile, join
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

import pandas as pd
import xarray as xr
from pyproj import CRS

from hydromt import hydromt_step
from hydromt._typing import DeferedFileClose, StrPath, XArrayDict
from hydromt._utils import _classproperty
from hydromt._utils.rgetattr import rgetattr
from hydromt._utils.steps_validator import validate_steps
from hydromt.components import (
    ModelComponent,
)
from hydromt.components.spatial import SpatialModelComponent
from hydromt.data_catalog import DataCatalog
from hydromt.gis.raster import GEO_MAP_COORD
from hydromt.plugins import PLUGINS
from hydromt.root import ModelRoot

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
    _MAPS = {"<general_hydromt_name>": "<model_name>"}
    _FOLDERS = [""]
    _TMP_DATA_DIR = None
    # supported model version should be filled by the plugins
    # e.g. _MODEL_VERSION = ">=1.0, <1.1"
    _MODEL_VERSION = None

    _API = {
        "maps": XArrayDict,
        "forcing": XArrayDict,
        "results": XArrayDict,
        "states": XArrayDict,
    }

    def __init__(
        self,
        *,
        components: Optional[Dict[str, Dict[str, Any]]] = None,
        root: Optional[str] = None,
        mode: str = "w",
        data_libs: Optional[Union[List, str]] = None,
        target_model_crs: Union[str, int] = 4326,
        logger=_logger,
        region_component: Optional[str] = None,
        **artifact_keys,
    ):
        r"""Initialize a model.

        Parameters
        ----------
        root : str, optional
            Model root, by default None
        mode : {'r','r+','w'}, optional
            read/append/write mode, by default "w"
        data_libs : List[str], optional
            List of data catalog configuration files, by default None
        \**artifact_keys:
            Additional keyword arguments to be passed down.
        logger:
            The logger to be used.
        """
        # Recursively update the options with any defaults that are missing in the configuration.
        components = components or {}
        self.target_crs = CRS.from_user_input(target_model_crs)

        data_libs = data_libs or []

        self.logger = logger

        # link to data
        self.data_catalog = DataCatalog(
            data_libs=data_libs, logger=self.logger, **artifact_keys
        )

        self._maps: Optional[XArrayDict] = None

        self._forcing: Optional[XArrayDict] = None
        self._states: Optional[XArrayDict] = None
        self._results: Optional[XArrayDict] = None

        # file system
        self.root: ModelRoot = ModelRoot(root or ".", mode=mode)

        self._components: Dict[str, ModelComponent] = {}
        self._add_components(components)

        self._defered_file_closes: List[DeferedFileClose] = []

        model_metadata = cast(
            Dict[str, str], PLUGINS.model_metadata[self.__class__.__name__]
        )
        self.logger.info(
            f"Initializing {self._NAME} model from {model_metadata['plugin_name']} (v{model_metadata['version']})."
        )

        self._region_component_name = self._determine_region_component(region_component)

    def _determine_region_component(self, region_component: Optional[str]) -> str:
        if region_component is not None:
            if region_component not in self._components:
                self.logger.warning(
                    f"Component {region_component} not found in components."
                    "You can add it afterwards with add_component."
                )
            elif not isinstance(
                self._components.get(region_component, None), SpatialModelComponent
            ):
                raise ValueError(
                    f"Component {region_component} is not a {SpatialModelComponent.__name__}."
                )
            return region_component
        else:
            has_region_components = [
                (name, c)
                for name, c in self._components.items()
                if isinstance(c, SpatialModelComponent)
            ]
            if len(has_region_components) > 1:
                raise ValueError(
                    "Multiple region components found in components. "
                    "Specify region_component."
                )
            if len(has_region_components) == 0:
                self.logger.warning("No region component found in components.")
                return ""
            return has_region_components[0][0]

    def _add_components(self, components: Dict[str, Dict[str, Any]]) -> None:
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
    def region(self) -> SpatialModelComponent:
        """Return the model's region component."""
        return cast(
            SpatialModelComponent, self._components[self._region_component_name]
        )

    @_classproperty
    def api(cls) -> Dict:
        """Return all model components and their data types."""
        _api = cls._API.copy()

        # reversed is so that child attributes take priority
        # this does mean that it becomes important in which order you
        # inherit from your base classes.
        for base_cls in reversed(cls.__mro__):
            if hasattr(base_cls, "_API"):
                _api.update(getattr(base_cls, "_API", {}))
        return _api

    def build(
        self,
        *,
        write: Optional[bool] = True,
        steps: List[Dict[str, Dict[str, Any]]],
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
        write: bool, optional
            Write complete model after executing all methods in opt, by default True.
        steps: Optional[List[Dict[str, Dict[str, Any]]]]
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
        steps: Optional[List[Dict[str, Dict[str, Any]]]] = None,
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
        steps: Optional[List[Dict[str, Dict[str, Any]]]]
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
        if self.region.region is None:
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
    def _options_contain_write(steps: List[Dict[str, Dict[str, Any]]]) -> bool:
        return any(
            next(iter(step_dict)).split(".")[-1] == "write" for step_dict in steps
        )

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

    def test_equal(self, other: "Model") -> tuple[bool, Dict[str, str]]:
        """Test if two models are equal, based on their components.

        Parameters
        ----------
        other : Model
            The model to compare against.

        Returns
        -------
        Tuple[bool, Dict[str, str]]
            True if equal, dictionary with errors per model component which is not equal.
        """
        if not isinstance(other, self.__class__):
            return False, {
                "__class__": f"f{other.__class__} does not inherit from {self.__class__}."
            }
        components = list(self._components.keys())
        components_other = list(other._components.keys())
        if components != components_other:
            return False, {
                "components": f"Components do not match: {components} != {components_other}"
            }

        errors: Dict[str, str] = {}
        is_equal = True
        for name, c in self._components.items():
            component_equal, component_errors = c.test_equal(other._components[name])
            is_equal &= component_equal
            errors.update(**component_errors)
        return is_equal, errors

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
        return self.target_crs


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
