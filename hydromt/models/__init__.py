# -*- coding: utf-8 -*-
"""HydroMT models API"""
import sys
from .model_api import Model
from . import model_plugins


# dictionary with entry points (not yet loaded!)
ENTRYPOINTS = model_plugins.discover()

# only load when requested
def __getattr__(name):
    name2class = {ep.object_name: name for name, ep in ENTRYPOINTS.items()}
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
    elif name in name2class:
        model_class = model_plugins.load(ENTRYPOINTS[name2class[name]])
        return model_class
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
