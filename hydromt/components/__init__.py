"""Implementations of the core ModelComponents."""

from hydromt.components.base import ModelComponent
from hydromt.components.config import ConfigComponent
from hydromt.components.datasets import DatasetsComponent
from hydromt.components.geoms import GeomsComponent
from hydromt.components.grid import GridComponent
from hydromt.components.mesh import MeshComponent
from hydromt.components.spatial import SpatialModelComponent
from hydromt.components.spatialdatasets import SpatialDatasetsComponent
from hydromt.components.tables import TablesComponent
from hydromt.components.vector import VectorComponent

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
