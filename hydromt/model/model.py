# -*- coding: utf-8 -*-
"""General and basic API for models in HydroMT."""

import logging
import os
import typing
from abc import ABCMeta
from inspect import _empty, signature
from os.path import isabs, isfile, join
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

import geopandas as gdp
from pyproj import CRS

from hydromt._typing import StrPath
from hydromt._typing.type_def import DeferedFileClose
from hydromt._utils import _rgetattr
from hydromt._utils.steps_validator import _validate_steps
from hydromt.data_catalog import DataCatalog
from hydromt.model import hydromt_step
from hydromt.model.components import (
    DatasetsComponent,
    ModelComponent,
    SpatialModelComponent,
)
from hydromt.model.root import ModelRoot
from hydromt.plugins import PLUGINS

__all__ = ["Model"]

_logger = logging.getLogger(__name__)
T = TypeVar("T", bound=ModelComponent)


class Model(object, metaclass=ABCMeta):
    """General and basic API for models in HydroMT."""

    name: str = "model"
    # supported model version should be filled by the plugins
    # e.g. _MODEL_VERSION = ">=1.0, <1.1"
    _MODEL_VERSION = None

    def __init__(
        self,
        root: Optional[str] = None,
        *,
        components: Optional[Dict[str, Any]] = None,
        mode: str = "w",
        data_libs: Optional[Union[List, str]] = None,
        region_component: Optional[str] = None,
        logger=_logger,
        **catalog_keys,
    ):
        """Initialize a model.

        Note that the * in the signature signifies that all of the arguments after the
        * in this function MUST be provided as keyword arguments.

        Parameters
        ----------
        root : str, optional
            Model root, by default None
        components: Dict[str, Any], optional
            Dictionary of components to add to the model, by default None
            Every entry in this dictionary contains the name of the component as key,
            and the component object as value, or a dictionary with options passed to the component initializers.
            If a component is a dictionary, the key 'type' should be provided with the name of the component type.
            .. code-block:: python
                {
                    "grid": {
                        "type": "GridComponent",
                        "filename": "path/to/grid.nc"
                    }
                }
        mode : {'r','r+','w'}, optional
            read/append/write mode, by default "w"
        data_libs : List[str], optional
            List of data catalog configuration files, by default None
        region_component : str, optional
            The name of the region component in the components dictionary.
            If None, the model will can automatically determine the region component if there is only one `SpatialModelComponent`.
            Otherwise it will raise an error.
            If there are no `SpatialModelComponent` it will raise a warning that `region` functionality will not work.
        logger:
            The logger to be used.
        **catalog_keys:
            Additional keyword arguments to be passed down to the DataCatalog.
        """
        # Recursively update the options with any defaults that are missing in the configuration.
        components = components or {}

        data_libs = data_libs or []

        self.logger = logger

        # link to data
        self.data_catalog = DataCatalog(
            data_libs=data_libs, logger=self.logger, **catalog_keys
        )

        # file system
        self.root: ModelRoot = ModelRoot(root or ".", mode=mode)

        self.components: Dict[str, ModelComponent] = {}
        self._add_components(components)

        self._defered_file_closes: List[DeferedFileClose] = []

        model_metadata = cast(Dict[str, str], PLUGINS.model_metadata[self.name])
        self.logger.info(
            f"Initializing {self.name} model from {model_metadata['plugin_name']} (v{model_metadata['version']})."
        )

        self._region_component_name = self._determine_region_component(region_component)

    def _determine_region_component(self, region_component: Optional[str]) -> str:
        if region_component is not None:
            if region_component not in self.components:
                raise KeyError(f"Component {region_component} not found in components.")
            elif not isinstance(
                self.components.get(region_component, None), SpatialModelComponent
            ):
                raise ValueError(
                    f"Component {region_component} is not a {SpatialModelComponent.__name__}."
                )
            return region_component
        else:
            has_region_components = [
                (name, c)
                for name, c in self.components.items()
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

    def _add_components(self, components: Dict[str, Any]) -> None:
        """Add all components that are specified in the config file."""
        for name, value in components.items():
            if isinstance(value, ModelComponent):
                self.add_component(name, value)
            elif isinstance(value, dict):
                type_name = value.pop("type")
                component_type = PLUGINS.component_plugins[type_name]
                self.add_component(name, component_type(self, **value))
            else:
                raise ValueError(
                    "Error in components argument. Type is not a ModelComponent or dict."
                )

    def add_component(self, name: str, component: ModelComponent) -> None:
        """Add a component to the model. Will raise an error if the component already exists."""
        if name in self.components:
            raise ValueError(f"Component {name} already exists in the model.")
        if not name.isidentifier():
            raise ValueError(f"Component name {name} is not a valid identifier.")
        self.components[name] = component

    def get_component(self, name: str) -> ModelComponent:
        """Get a component from the model. Will raise an error if the component does not exist."""
        return self.components[name]

    def __getattr__(self, name: str) -> ModelComponent:
        """Get a component from the model. Will raise an error if the component does not exist."""
        return self.get_component(name)

    @property
    def region(self) -> Optional[gdp.GeoDataFrame]:
        """Return the model's region component."""
        return (
            cast(
                SpatialModelComponent, self.components[self._region_component_name]
            ).region
            if self._region_component_name in self.components
            else None
        )

    @property
    def crs(self) -> Optional[CRS]:
        """Returns coordinate reference system embedded in region."""
        return self.region.crs if self.region is not None else None

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
        _validate_steps(self, steps)

        for step_dict in steps:
            step, kwargs = next(iter(step_dict.items()))
            self.logger.info(f"build: {step}")
            # Call the methods.
            method = _rgetattr(self, step)
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
        _validate_steps(self, steps)

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
        if self.region is None:
            raise ValueError("Model region not found, setup model using `build` first.")

        # loop over methods from config file
        for step_dict in steps:
            step, kwargs = next(iter(step_dict.items()))
            self.logger.info(f"update: {step}")
            # Call the methods.
            method = _rgetattr(self, step)
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

        for comp in self.components.values():
            if isinstance(comp, DatasetsComponent):
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
        components = components or list(self.components.keys())
        for c in [self.components[name] for name in components]:
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
        components = components or list(self.components.keys())
        for c in [self.components[name] for name in components]:
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
        for name, source in self.data_catalog.list_sources(used_only=used_only):
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
        components = list(self.components.keys())
        components_other = list(other.components.keys())
        if components != components_other:
            return False, {
                "components": f"Components do not match: {components} != {components_other}"
            }

        errors: Dict[str, str] = {}
        is_equal = True
        for name, c in self.components.items():
            component_equal, component_errors = c.test_equal(other.components[name])
            is_equal &= component_equal
            errors.update(**component_errors)
        return is_equal, errors


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
