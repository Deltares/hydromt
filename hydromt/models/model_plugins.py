"""Implementation of the mechanism to access the plugin entrypoints."""
import logging
from typing import Dict, Iterator, List

from entrypoints import Distribution, EntryPoint, get_group_all

from .. import __version__, _compat
from .model_api import Model

logger = logging.getLogger(__name__)

__all__ = ["ModelCatalog"]

# local generic models
LOCAL_EPS = {
    "grid_model": "hydromt.models.model_grid:GridModel",
    "vector_model": "hydromt.models.model_vector:VectorModel",
    "mesh_model": "hydromt.models.model_mesh:MeshModel",
    "network_model": "hydromt.models.model_network:NetworkModel",
}


def get_general_eps() -> Dict:
    """Get local hydromt generic model class entrypoints.

    Returns
    -------
    eps : dict
        Entrypoints dict
    """
    eps = {}
    distro = Distribution("hydromt", __version__)
    for name, epstr in LOCAL_EPS.items():
        if name == "mesh_model" and not _compat.HAS_XUGRID:
            continue
        eps[name] = EntryPoint.from_string(epstr, name, distro)
    return eps


def _discover(path=None) -> List:
    """Discover drivers via entrypoints."""
    return get_group_all("hydromt.models", path=path)


def get_plugin_eps(path=None, logger=logger) -> Dict:
    """Discover hydromt model plugins based on 'hydromt.models' entrypoints.

    Parameters
    ----------
    path : str or None
        Default is ``sys.path``.
    logger : logger object, optional
        The logger object used for logging messages. If not provided, the default
        logger will be used.

    Returns
    -------
    eps : dict
        Entrypoints dict
    """
    eps = {}
    for ep in _discover():
        name = ep.name
        if name in eps or name in LOCAL_EPS:
            plugin = f"{ep.module_name}.{ep.object_name}"
            logger.warning(f"Duplicated model plugin '{name}'; skipping {plugin}")
            continue
        logger.debug(
            f"Discovered model plugin '{name} = {ep.module_name}.{ep.object_name}' "
            f"({ep.distro.version})"
        )
        eps[ep.name] = ep
    return eps


def load(ep, logger=logger) -> Model:
    """Load entrypoint and return plugin model class.

    Parameters
    ----------
    ep : entrypoint
        discovered entrypoint
    logger : logger object, optional
        The logger object used for logging messages. If not provided, the default
        logger will be used.

    Returns
    -------
    model_class : Model
        plugin model class
    """
    _str = f"{ep.name} = {ep.module_name}.{ep.object_name}"
    try:
        model_class = ep.load()
        if not issubclass(model_class, Model):
            raise ValueError(f"Model plugin type not recognized '{_str}'")
        logger.debug(f"Loaded model plugin {_str}")
        return model_class
    except (ModuleNotFoundError, AttributeError) as err:
        raise ImportError(f"Error while loading model plugin '{_str}' ({err})")


class ModelCatalog:

    """The model catalogue provides access to plugins and their Model classes."""

    def __init__(self):
        """Initiate the catalog object."""
        self._eps = {}  # entrypoints
        self._cls = {}  # classes
        self._plugins = []  # names of plugins
        self._general = []  # names of local model classes

    @property
    def eps(self) -> Dict:
        """Return dictionary with available model entrypoints."""
        if len(self._eps) == 0:
            _ = self.plugins  # discover plugins
            _ = self.generic  # get generic local model classes
        return self._eps

    @property
    def cls(self) -> Dict:
        """Return dictionary with available model classes."""
        if len(self._cls) != len(self.eps):
            for name in self.eps:
                if name not in self._cls:
                    self._cls[name] = load(self.eps[name])
        return self._cls

    @property
    def plugins(self) -> List:
        """Return list with names of model plugins."""
        if len(self._plugins) == 0:
            eps = get_plugin_eps()
            self._plugins = list(eps.keys())
            self._eps.update(**eps)
        return self._plugins

    @property
    def generic(self) -> List:
        """Return list with names of generic models."""
        if len(self._general) == 0:
            eps = get_general_eps()
            self._general = list(eps.keys())
            self._eps.update(**eps)
        return self._general

    def load(self, name) -> Model:
        """Return model class."""
        if name not in self._cls:
            self._cls[name] = load(self[name])
        return self._cls[name]

    def __str__(self):
        """Generate string representation containing the registered entrypoints."""
        plugins = "".join(
            [
                f" - {name} ({self.eps[name].distro.name}"
                f" {self.eps[name].distro.version})\n"
                for name in self.plugins
            ]
        )
        generic = "".join([f" - {name}\n" for name in self.generic])
        return (
            f"model plugins:\n{plugins}generic models (hydromt {__version__})"
            f":\n{generic}"
        )

    def __getitem__(self, name) -> Model:
        """Return the entrypoint with the provided name."""
        if name not in self.eps:
            raise ValueError(f"Unknown model {name}; select from {self.eps.keys()}")
        return self._eps[name]

    def __iter__(self) -> Iterator:
        """Return an iterator over registered entrypoints."""
        return iter(self.eps)
