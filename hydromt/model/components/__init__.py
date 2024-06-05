"""Implementations of the core ModelComponents."""

from .base import ModelComponent
from .config import ConfigComponent
from .datasets import DatasetsComponent
from .geoms import GeomsComponent
from .grid import GridComponent
from .mesh import MeshComponent
from .spatial import SpatialModelComponent
from .spatialdatasets import SpatialDatasetsComponent
from .tables import TablesComponent
from .vector import VectorComponent

__all__ = [
    "ConfigComponent",
    "GeomsComponent",
    "GridComponent",
    "ModelComponent",
    "TablesComponent",
    "VectorComponent",
    "MeshComponent",
    "DatasetsComponent",
    "SpatialModelComponent",
    "SpatialDatasetsComponent",
]

# define hydromt component entry points; abstract classes are not included
# see also hydromt.component group in pyproject.toml
__hydromt_eps__ = [
    "ConfigComponent",
    "GeomsComponent",
    "GridComponent",
    "TablesComponent",
    "VectorComponent",
    "MeshComponent",
    "DatasetsComponent",
]
