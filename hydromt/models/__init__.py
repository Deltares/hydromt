# -*- coding: utf-8 -*-
"""HydroMT models API"""
import sys

from .model_api import *
from .model_grid import *
from .model_lumped import *
from .model_network import *
from . import model_plugins

from .. import _has_xugrid

if _has_xugrid():
    from .model_mesh import *

# dictionary with entry points (not yet loaded!)
ENTRYPOINTS = model_plugins.discover()
PLUGINS = {ep.object_name: name for name, ep in ENTRYPOINTS.items()}

# only load when requested
def __getattr__(name):
    thismodule = sys.modules[__name__]

    # load a register all models
    if name == "MODELS":
        MODELS = {
            ep.name: model_plugins.load(ep, thismodule) for ep in ENTRYPOINTS.values()
        }
        setattr(thismodule, name, MODELS)
        return MODELS

    # trick to allow import of plugin model class from hydromt core
    # from hydromt.models import xxxxModel
    elif name in PLUGINS:
        model_class = model_plugins.load(ENTRYPOINTS[PLUGINS[name]])
        return model_class
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
