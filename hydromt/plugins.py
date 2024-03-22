"""Implementation of the mechanism to access the plugin entrypoints."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type, cast

from importlib_metadata import entry_points

if TYPE_CHECKING:
    from hydromt.components import ModelComponent  # noqa
    from hydromt.models import Model  # noqa

__all__ = ["PLUGINS"]


def _discover_plugins(group: str) -> Tuple[Dict[str, Type], Dict[str, Dict[str, str]]]:
    plugins: Dict[str, Type] = {}
    plugin_metadata: Dict[str, Dict[str, str]] = {}
    eps = entry_points(group=group)
    for ep in eps:
        module = ep.load()
        for attr_name in module.__all__:
            attr = getattr(module, attr_name)
            if attr_name not in plugins:
                plugins[attr_name] = attr

                # this is for display only, hence string
                plugin_metadata[attr_name] = {
                    "name": str(ep.dist.name),
                    "object": attr_name,
                    "version": str(ep.dist.version),
                }
            else:
                raise ValueError(f"Conflicting definitions for component {attr_name}")

    return plugins, plugin_metadata


def _format_metadata(d: Dict[str, str]):
    return "{object} ({name} {version})".format(
        object=d["object"], name=d["name"], version=d["version"]
    )


class Plugins:
    """The model catalogue provides access to plugins."""

    def __init__(self):
        """Initiate the catalog object."""
        self._component_plugins: Optional[dict[str, type["ModelComponent"]]] = None
        self._model_plugins: Optional[dict[str, type["Model"]]] = None
        self._component_metadata: Optional[Dict[str, Dict[str, str]]] = None
        self._model_metadata: Optional[Dict[str, Dict[str, str]]] = None

    def _initialize_component_plugins(self) -> None:
        self._component_plugins, new_metadata = _discover_plugins(
            group="hydromt.components"
        )

        if self._component_metadata is None:
            self._component_metadata = new_metadata
        else:
            self._component_metadata = {**self._component_metadata, **new_metadata}

    @property
    def component_plugins(self) -> dict[str, type["ModelComponent"]]:
        """Load and provide access to all known component plugins."""
        if self._component_plugins is None:
            self._initialize_component_plugins()

        if self._component_plugins is None:
            # core itself exposes plugins so if we can't find anything, something is wrong
            raise RuntimeError("Could not load any component plugins")
        else:
            return self._component_plugins

    def _initialise_model_plugins(self) -> None:
        self._model_plugins, new_metadata = _discover_plugins(group="hydromt.models")

        if self._model_metadata is None:
            self._model_metadata = new_metadata
        else:
            self._model_metadata = {**self._model_metadata, **new_metadata}

    @property
    def model_plugins(self) -> dict[str, type["Model"]]:
        """Load and provide access to all known model plugins."""
        if self._model_plugins is None:
            self._initialise_model_plugins()

        if self._model_plugins is None:
            # core itself exposes plugins so if we can't find anything, something is wrong
            raise RuntimeError("Could not load any model plugins")
        else:
            return self._model_plugins

    def model_summary(self) -> str:
        """Generate string representation containing the registered model entrypoints."""
        self._initialise_model_plugins()
        self._model_metadata = cast(Dict[str, Dict[str, str]], self._model_metadata)
        model_plugins = "\n\t- ".join(
            map(_format_metadata, self._model_metadata.values())
        )
        return f"Model plugins:\n\t- {model_plugins}"

    def component_summary(self) -> str:
        """Generate string representation containing the registered component entrypoints."""
        self._initialize_component_plugins()
        self._component_metadata = cast(
            Dict[str, Dict[str, str]], self._component_metadata
        )
        component_plugins = "\n\t- ".join(
            map(_format_metadata, self._component_metadata.values())
        )
        return f"Component plugins:\n\t- {component_plugins}"

    def plugin_summary(self) -> str:
        return "\n".join([self.model_summary(), self.component_summary()])


PLUGINS = Plugins()
