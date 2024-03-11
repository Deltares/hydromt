# -*- coding: utf-8 -*-
"""HydroMT models API."""
from hydromt._compat import HAS_XUGRID
from hydromt.models.api import Model
from hydromt.models.components.grid import GridComponent
from hydromt.models.components.network import NetworkModel
from hydromt.models.components.region import ModelRegionComponent
from hydromt.models.components.vector import VectorModel
from hydromt.models.plugins import ModelCatalog

if HAS_XUGRID:
    from hydromt.models.components.mesh import MeshModel

from hydromt.models.root import ModeLike, ModelMode, ModelRoot

# expose global MODELS object which discovers and loads
# any local generalized or plugin model class on-the-fly
MODELS = ModelCatalog()
