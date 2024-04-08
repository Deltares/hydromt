"""Implementation of the mechanism to access the plugin entrypoints."""

import inspect
from typing import TYPE_CHECKING, Dict, Optional, Type, TypedDict, cast

from importlib_metadata import entry_points

if TYPE_CHECKING:
    from hydromt.components import ModelComponent  # noqa
    from hydromt.drivers import BaseDriver
    from hydromt.models import Model  # noqa

__all__ = ["PLUGINS"]

Plugin = TypedDict("Plugin", {"plugin_name": str, "type": Type, "version": str})


def _discover_plugins(group: str) -> Dict[str, Plugin]:
    plugins: Dict[str, Plugin] = {}
    eps = entry_points(group=group)
    for ep in eps:
        module = ep.load()
        for attr_name in module.__all__:
            attr = getattr(module, attr_name)
            if attr_name not in plugins:
                # this is for display only, hence string
                plugins[attr_name] = {
                    "plugin_name": str(ep.dist.name),
                    "type": attr,
                    "version": str(ep.dist.version),
                }
            else:
                raise ValueError(f"Conflicting definitions for component {attr_name}")

    return plugins


def _format_metadata(metadata: Dict[str, str]) -> str:
    return "{type} ({plugin_name} {version})".format(
        type=metadata["type"].__name__,  # type: ignore
        plugin_name=metadata["plugin_name"],
        version=metadata["version"],
    )


class Plugins:
    """The model catalogue provides access to plugins."""

    def __init__(self) -> None:
        """Initiate the catalog object."""
        self._component_plugins: Optional[Dict[str, Plugin]] = None
        self._driver_plugins: Optional[Dict[str, Plugin]] = None
        self._model_plugins: Optional[Dict[str, Plugin]] = None

    def _initialize_plugins(self) -> None:
        self._component_plugins = _discover_plugins(group="hydromt.components")
        self._driver_plugins = _discover_plugins(group="hydromt.drivers")
        self._model_plugins = _discover_plugins(group="hydromt.models")

    @property
    def component_plugins(self) -> dict[str, type["ModelComponent"]]:
        """Load and provide access to all known component plugins."""
        if self._component_plugins is None:
            self._initialize_plugins()

        if self._component_plugins is None:
            # core itself exposes plugins so if we can't find anything, something is wrong
            raise RuntimeError("Could not load any component plugins")
        else:
            return cast(
                Dict[str, Type["ModelComponent"]],
                {
                    name: value["type"]
                    for name, value in self._component_plugins.items()
                },
            )

    @property
    def driver_plugins(self) -> dict[str, Type["BaseDriver"]]:
        """Load and provide access to all known driver plugins."""
        if self._driver_plugins is None:
            self._initialize_plugins()

        if self._driver_plugins is None:
            # core itself exposes plugins so if we can't find anything, something is wrong
            raise RuntimeError("Could not load any driver plugins")

        return {
            name: value["type"]
            for name, value in self._driver_plugins.items()
            if not inspect.isabstract(value["type"])
        }

    @property
    def model_plugins(self) -> dict[str, type["Model"]]:
        """Load and provide access to all known model plugins."""
        if self._model_plugins is None:
            self._initialize_plugins()

        if self._model_plugins is None:
            # core itself exposes plugins so if we can't find anything, something is wrong
            raise RuntimeError("Could not load any model plugins")
        else:
            return cast(
                Dict[str, Type["Model"]],
                {name: value["type"] for name, value in self._model_plugins.items()},
            )

    @property
    def model_metadata(self) -> Dict[str, Dict[str, str]]:
        """Load and provide access to all known model plugins."""
        if self._model_plugins is None:
            self._initialize_plugins()

        if self._model_plugins is None:
            # core itself exposes plugins so if we can't find anything, something is wrong
            raise RuntimeError("Could not load any model plugins")
        else:
            return cast(
                Dict[str, Dict[str, str]],
                {k: v for k, v in self._model_plugins.items() if k != "type"},
            )

    @property
    def component_metadata(self) -> Dict[str, Dict[str, str]]:
        """Load and provide access to all known model plugins."""
        if self._component_plugins is None:
            self._initialize_plugins()

        if self._component_plugins is None:
            # core itself exposes plugins so if we can't find anything, something is wrong
            raise RuntimeError("Could not load any model plugins")
        else:
            return cast(
                Dict[str, Dict[str, str]],
                {k: v for k, v in self._component_plugins.items() if k != "type"},
            )

    @property
    def driver_metadata(self) -> Dict[str, Dict[str, str]]:
        """Load and provide access to all known driver plugin metadata."""
        if self._driver_plugins is None:
            self._initialize_plugins()

        if self._driver_plugins is None:
            # core itself exposes plugins so if we can't find anything, something is wrong
            raise RuntimeError("Could not load any driver plugins")
        else:
            return cast(
                Dict[str, Dict[str, str]],
                {
                    k: v
                    for k, v in self._driver_plugins.items()
                    if not inspect.isabstract(v["type"])  # filter ABCs from core
                },
            )

    def model_summary(self) -> str:
        """Generate string representation containing the registered model entrypoints."""
        s = ""
        for metadata in self.model_metadata.values():
            s += f"\n\t- {_format_metadata(metadata)}"
        model_plugins = "\n\t- ".join(
            map(_format_metadata, self.model_metadata.values())
        )
        return f"Model plugins:\n\t- {model_plugins}"

    def driver_summary(self) -> str:
        """Generate string representation container the registered driver entrypoints."""
        self._initialize_plugins()
        driver_plugins = "\n\t ".join(
            list(map(_format_metadata, self.driver_metadata.values()))
        )
        return f"Driver Plugins:\n\t- {driver_plugins}"

    def component_summary(self) -> str:
        """Generate string representation containing the registered component entrypoints."""
        self._initialize_plugins()
        component_plugins = "\n\t- ".join(
            list(map(_format_metadata, self.component_metadata.values()))
        )
        return f"Component plugins:\n\t- {component_plugins}"

    def plugin_summary(self) -> str:
        return "\n".join([self.model_summary(), self.component_summary()])


PLUGINS = Plugins()
