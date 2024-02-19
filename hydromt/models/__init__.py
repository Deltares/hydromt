# -*- coding: utf-8 -*-
"""HydroMT models API."""
from hydromt.models.api import Model
from hydromt.models.components.grid import GridModel
from hydromt.models.components.mesh import MeshModel
from hydromt.models.components.network import NetworkModel
from hydromt.models.components.vector import VectorModel
from hydromt.models.plugins import ModelCatalog

# expose global MODELS object which discovers and loads
# any local generalized or plugin model class on-the-fly
MODELS = ModelCatalog()
