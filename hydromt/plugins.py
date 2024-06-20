"""Implementation of the mechanism to access the plugin entrypoints."""

import inspect
from abc import abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Type, TypedDict, cast

from importlib_metadata import entry_points

if TYPE_CHECKING:
    from hydromt.data_catalog.drivers import BaseDriver
    from hydromt.data_catalog.predefined_catalog import PredefinedCatalog
    from hydromt.model import Model
    from hydromt.model.components import ModelComponent

__all__ = ["PLUGINS"]

Plugin = TypedDict(
    "Plugin", {"type": Type, "name": str, "plugin_name": str, "version": str}
)


def _format_metadata(metadata: Dict[str, str]) -> str:
    return "{name} ({plugin_name} {version})".format(
        name=metadata["name"],
        plugin_name=metadata["plugin_name"],
        version=metadata["version"],
    )


class PluginGroup:
    group: ClassVar[str] = None
    base_module: ClassVar[str] = None
    base_class: ClassVar[str] = None

    def __init__(self) -> None:
        self._plugins: Optional[Dict[str, Plugin]] = None

    def _initialize_plugins(self) -> None:
        # load the base class to check for subclasses
        # not pretty, but we import dynamically to avoid circular imports
        mod = import_module(self.base_module)
        base_class = getattr(mod, self.base_class)

        plugins: Dict[str, Plugin] = {}
        eps = entry_points(group=self.group)
        for ep in eps:
            module = ep.load()
            hydromt_eps = getattr(module, "__hydromt_eps__", None)
            ep_name = getattr(ep, "name", None)  # this cannot be mocked?
            if hydromt_eps is None or not isinstance(hydromt_eps, list):
                raise ValueError(
                    f"{self.group} plugin {ep_name} ({ep.dist.name}) does not define __hydromt_eps__ list attribute"
                )
            for attr_name in hydromt_eps:
                attr = getattr(module, attr_name, None)
                # check if the attribute is a subclass of the expected type and not an ABCs from core
                if (
                    not inspect.isclass(attr)
                    or inspect.isabstract(attr)
                    or not issubclass(attr, base_class)
                ):
                    raise ValueError(
                        f"{self.group} plugin {ep_name} {attr_name} ({ep.dist.name}) is not a valid {self.base_class}"
                    )
                name = getattr(attr, "name", attr_name)
                if name not in plugins:
                    # other than type, this is for display only, hence string
                    plugins[name] = {
                        "type": attr,
                        "name": name,
                        "plugin_name": str(ep.dist.name),
                        "version": str(ep.dist.version),
                    }
                else:
                    raise ValueError(f"Conflicting definitions for {self.group} {name}")

        # we should have at least one plugin from core
        if not plugins:
            raise RuntimeError(f"Could not load any {self.group} plugins")

        self._plugins = plugins

    @property
    @abstractmethod
    def plugins(self) -> dict[str, Type]:
        pass

    @property
    def metadata(self) -> Dict[str, Dict[str, str]]:
        if self._plugins is None:
            self._initialize_plugins()

        return cast(
            Dict[str, Dict[str, str]],
            {k: v for k, v in self._plugins.items() if isinstance(k, str)},
        )

    def summary(self) -> str:
        name = self.group.split(".")[-1][:-1].capitalize()
        s = ""
        for metadata in self.metadata.values():
            s += f"\n\t- {_format_metadata(metadata)}"
        plugins = "\n\t- ".join(map(_format_metadata, self.metadata.values()))
        return f"{name} plugins:\n\t- {plugins}"


class ComponentPlugins(PluginGroup):
    group = "hydromt.components"
    base_module = "hydromt.model.components"
    base_class = "ModelComponent"

    @property
    def plugins(self) -> dict[str, Type["ModelComponent"]]:
        if self._plugins is None:
            self._initialize_plugins()

        return cast(
            Dict[str, Type["ModelComponent"]],
            {name: value["type"] for name, value in self._plugins.items()},
        )


class DriverPlugins(PluginGroup):
    group = "hydromt.drivers"
    base_module = "hydromt.data_catalog.drivers"
    base_class = "BaseDriver"

    @property
    def plugins(self) -> dict[str, Type["BaseDriver"]]:
        if self._plugins is None:
            self._initialize_plugins()

        return cast(
            Dict[str, Type["BaseDriver"]],
            {name: value["type"] for name, value in self._plugins.items()},
        )


class ModelPlugins(PluginGroup):
    group = "hydromt.models"
    base_module = "hydromt.model.model"
    base_class = "Model"

    @property
    def plugins(self) -> dict[str, Type["Model"]]:
        if self._plugins is None:
            self._initialize_plugins()

        return cast(
            Dict[str, Type["Model"]],
            {name: value["type"] for name, value in self._plugins.items()},
        )


class CatalogPlugins(PluginGroup):
    group = "hydromt.catalogs"
    base_module = "hydromt.data_catalog.predefined_catalog"
    base_class = "PredefinedCatalog"

    @property
    def plugins(self) -> dict[str, Type["PredefinedCatalog"]]:
        if self._plugins is None:
            self._initialize_plugins()

        return cast(
            Dict[str, Type["PredefinedCatalog"]],
            {name: value["type"] for name, value in self._plugins.items()},
        )


class Plugins:
    """The model catalogue provides access to plugins."""

    def __init__(self) -> None:
        """Initiate the catalog object."""
        self._component_plugins: ComponentPlugins = ComponentPlugins()
        self._driver_plugins: DriverPlugins = DriverPlugins()
        self._model_plugins: ModelPlugins = ModelPlugins()
        self._catalog_plugins: CatalogPlugins = CatalogPlugins()

    @property
    def component_plugins(self) -> dict[str, type["ModelComponent"]]:
        """Load and provide access to all known model component plugins."""
        return self._component_plugins.plugins

    @property
    def driver_plugins(self) -> dict[str, Type["BaseDriver"]]:
        """Load and provide access to all known driver plugins."""
        return self._driver_plugins.plugins

    @property
    def model_plugins(self) -> dict[str, type["Model"]]:
        """Load and provide access to all known model plugins."""
        return self._model_plugins.plugins

    @property
    def catalog_plugins(self) -> dict[str, Type["PredefinedCatalog"]]:
        """Load and provide access to all known catalog plugins."""
        return self._catalog_plugins.plugins

    @property
    def model_metadata(self) -> Dict[str, Dict[str, str]]:
        """Load and provide access to all known model plugins."""
        return self._model_plugins.metadata

    @property
    def component_metadata(self) -> Dict[str, Dict[str, str]]:
        """Load and provide access to all known model component plugins."""
        return self._component_plugins.metadata

    @property
    def driver_metadata(self) -> Dict[str, Dict[str, str]]:
        """Load and provide access to all known driver plugin metadata."""
        return self._driver_plugins.metadata

    @property
    def catalog_metadata(self) -> Dict[str, Dict[str, str]]:
        """Load and provide access to all known catalog plugin metadata."""
        return self._catalog_plugins.metadata

    def model_summary(self) -> str:
        """Generate string representation containing the registered model entrypoints."""
        return self._model_plugins.summary()

    def driver_summary(self) -> str:
        """Generate string representation container the registered driver entrypoints."""
        return self._driver_plugins.summary()

    def component_summary(self) -> str:
        """Generate string representation containing the registered model component entrypoints."""
        return self._component_plugins.summary()

    def catalog_summary(self) -> str:
        """Generate string representation containing the registered catalog entrypoints."""
        return self._catalog_plugins.summary()

    def plugin_summary(self) -> str:
        return "\n".join(
            [
                self.model_summary(),
                self.component_summary(),
                self.driver_summary(),
                self.catalog_summary(),
            ]
        )


PLUGINS = Plugins()
