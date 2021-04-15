# -*- coding: utf-8 -*-
"""HydroMT models API"""
import sys
from .model_api import Model
from . import model_plugins

# model dictionary to be used in command line interface
MODELS = {}

MODELS.update(model_plugins.discover())

# make models available for import
# from hydromt.models import xxxxModel
thismodule = sys.modules[__name__]
for model_class in MODELS.values():
    setattr(thismodule, model_class.__name__, model_class)
