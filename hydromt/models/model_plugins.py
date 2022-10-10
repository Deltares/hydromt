from entrypoints import EntryPoint, Distribution, get_group_all
import logging
from typing import Dict, Iterator, List
from .. import __version__
from .model_api import Model
from .. import _compat

logger = logging.getLogger(__name__)

__all__ = ["ModelCatalog"]

# local generic models
LOCAL_EPS = {
    "grid_model": "hydromt.models.model_grid:GridModel",
    "lumped_model": "hydromt.models.model_lumped:LumpedModel",
    "mesh_model": "hydromt.models.model_mesh:MeshModel",
    "network_model": "hydromt.models.model_network:NetworkModel",
}


def get_general_eps() -> Dict:
    """Get local hydromt generic model class entrypoints

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
    """Discover drivers via entrypoints"""
    return get_group_all("hydromt.models", path=path)


def get_plugin_eps(path=None, logger=logger) -> Dict:
    """Discover hydromt model plugins based on 'hydromt.models' entrypoints

    Parameters
    ----------
    path : str or None
        Default is ``sys.path``.

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
            f"Discovered model plugin '{name} = {ep.module_name}.{ep.object_name}' ({ep.distro.version})"
        )
        eps[ep.name] = ep
    return eps


def load(ep, logger=logger) -> Model:
    """Load entrypoint and return plugin model class

    Parameters
    ----------
    ep : entrypoint
        discovered entrypoint

    Returns
    -------
    model_class : Model
        plugin model class
    """
    _str = f"{ep.name} = {ep.module_name}.{ep.object_name}"
    try:
        # plugins[ep.name] = ep.load()
        model_class = ep.load()
        if module is not None:
            setattr(module, model_class.__name__, model_class)
        logger.debug(
            f"Loaded model plugin '{ep.name} = {ep.module_name}.{ep.object_name}' ({ep.distro.version})"
        )
        return model_class
    except (ModuleNotFoundError, AttributeError) as err:
        logger.exception(f"Error while loading entrypoint {ep.name}: {str(err)}")
        return None
