"""Implementations of the core ModelComponents."""

from hydromt.model.components.base import ModelComponent
from hydromt.model.components.config import ConfigComponent
from hydromt.model.components.datasets import DatasetsComponent
from hydromt.model.components.geoms import GeomsComponent
from hydromt.model.components.grid import GridComponent
from hydromt.model.components.mesh import MeshComponent
from hydromt.model.components.spatial import SpatialModelComponent
from hydromt.model.components.spatialdatasets import SpatialDatasetsComponent
from hydromt.model.components.tables import TablesComponent
from hydromt.model.components.vector import VectorComponent

__all__ = [
    "ConfigComponent",
    "DatasetsComponent",
    "GeomsComponent",
    "GridComponent",
    "MeshComponent",
    "ModelComponent",
    "SpatialDatasetsComponent",
    "SpatialModelComponent",
    "TablesComponent",
    "VectorComponent",
]

# define hydromt component entry points; abstract classes are not included
# see also hydromt.component group in pyproject.toml
__hydromt_eps__ = [
    "ConfigComponent",
    "DatasetsComponent",
    "GeomsComponent",
    "GridComponent",
    "MeshComponent",
    "SpatialDatasetsComponent",
    "TablesComponent",
    "VectorComponent",
]
