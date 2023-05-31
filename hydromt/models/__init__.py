# -*- coding: utf-8 -*-
"""HydroMT models API."""
from .. import _compat
from .model_api import Model
from .model_grid import GridMixin, GridModel
from .model_lumped import LumpedModel
from .model_network import NetworkModel
from .model_plugins import EntryPoint, ModelCatalog

__all__ = [
    "Model",
    "GridMixin",
    "GridModel",
    "LumpedModel",
    "NetworkModel",
    "EntryPoint",
    "ModelCatalog",
]

# NOTE: pygeos is still required in XUGRID;
# remove requirement after https://github.com/Deltares/xugrid/issues/33
if _compat.HAS_XUGRID and _compat.HAS_PYGEOS:
    from .model_mesh import MeshModel, MeshMixin

    __all__.append(MeshModel, MeshMixin)

# expose global MODELS object which discovers and loads
# any local generalized or plugin model class on-the-fly
MODELS = ModelCatalog()
