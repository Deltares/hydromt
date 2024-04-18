# -*- coding: utf-8 -*-
"""General and basic API for models in HydroMT."""

import logging
import os
import typing
from abc import ABCMeta
from inspect import _empty, signature
from os.path import isabs, isfile, join
from pathlib import Path
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
from hydromt._typing import StrPath
from hydromt._utils.rgetattr import rgetattr
from hydromt._utils.steps_validator import validate_steps
from hydromt.components import (
    ModelComponent,
    ModelRegionComponent,
)
from hydromt.components.dataset import DatasetComponent
from hydromt.data_catalog import DataCatalog
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

    _NAME: str = "modelname"
    # supported model version should be filled by the plugins
    # e.g. _MODEL_VERSION = ">=1.0, <1.1"
    _MODEL_VERSION = None

    def __init__(
        self,
        components: Optional[dict[str, dict[str, Any]]] = None,
        root: Optional[str] = None,
        mode: str = "w",
        data_libs: Optional[Union[List, str]] = None,
        target_model_crs: Union[str, int] = 4326,
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

        self.target_crs = CRS.from_user_input(target_model_crs)

        data_libs = data_libs or []

        self.logger = logger

        # link to data
        self.data_catalog = DataCatalog(
            data_libs=data_libs, logger=self.logger, **artifact_keys
        )

        # file system
        self.root: ModelRoot = ModelRoot(root or ".", mode=mode)

        self._components: Dict[str, ModelComponent] = {}
        self._add_components(components)

        self._defered_file_closes = []

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

        for comp in self._components.values():
            if isinstance(comp, DatasetComponent):
                comp._cleanup(forceful_overwrite=forceful_overwrite)

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

        errors: dict[str, str] = {}
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

    @property
    def crs(self) -> CRS:
        """Returns coordinate reference system embedded in region."""
        return self.target_crs


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
