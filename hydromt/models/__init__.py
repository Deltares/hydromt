# -*- coding: utf-8 -*-
"""HydroMT models API"""
from .model_api import *
from .model_grid import *
from .model_lumped import *
from .model_network import *
from .model_plugins import *
from .. import _compat

# NOTE: pygeos is still required in XUGRID;
# remove requirement after https://github.com/Deltares/xugrid/issues/33
if _compat.HAS_XUGRID and _compat.HAS_PYGEOS:
    from .model_mesh import MeshModel

# expose global MODELS object which discovers and loads
# any local generalized or plugin model class on-the-fly
MODELS = ModelCatalog()
