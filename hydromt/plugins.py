"""Implementation of the mechanism to access the plugin entrypoints."""

import logging
from typing import TYPE_CHECKING, Dict, Type

from importlib_metadata import entry_points

if TYPE_CHECKING:
    from hydromt.components import ModelComponent  # noqa
    from hydromt.models import Model  # noqa

logger = logging.getLogger(__name__)

__all__ = ["PLUGINS"]


def _discover_plugins(group: str) -> Dict[str, Type]:
    plugins = {}
    eps = entry_points(group=group)
    for ep in eps:
        module = ep.load()
        for attr_name in module.__all__:
            attr = getattr(module, attr_name)
            if attr_name not in plugins:
                plugins[attr_name] = attr
            else:
                raise ValueError(f"Conflicting definitions for component {attr_name}")

    return plugins


class Plugins:
    """The model catalogue provides access to plugins."""

    def __init__(self):
        """Initiate the catalog object."""
        self._component_plugins = None
        self._model_plugins = None

    @property
    def component_plugins(self):
        """Load and provide access to all known component plugins."""
        if self._component_plugins is None:
            self._component_plugins = _discover_plugins(group="hydromt.components")

        return self._component_plugins

    @property
    def model_plugins(self):
        """Load and provide access to all known model plugins."""
        if self._model_plugins is None:
            self._model_plugins = _discover_plugins(group="hydromt.models")

        return self._model_plugins

    def summary(self):
        """Generate string representation containing the registered entrypoints."""
        component_plugins = ", ".join(self.component_plugins)
        model_plugins = ", ".join(self.model_plugins)
        return (
            f"component plugins:\n{component_plugins}"
            f"model plugins:\n{model_plugins}"
        )


PLUGINS = Plugins()
