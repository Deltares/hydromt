"""Implementations of the core ModelComponents."""

from .base import ModelComponent
from .config import ConfigComponent
from .datasets import DatasetsComponent
from .geoms import GeomsComponent
from .grid import GridComponent, GridExtraComponent
from .mesh import MeshComponent, MeshComponentExtra
from .spatial import SpatialModelComponent
from .spatialdatasets import SpatialDatasetsComponent
from .tables import TablesComponent
from .vector import VectorComponent

__all__ = [
    "ConfigComponent",
    "DatasetsComponent",
    "GeomsComponent",
    "GridComponent",
    "GridExtraComponent",
    "MeshComponent",
    "MeshComponentExtra",
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
    "GridExtraComponent",
    "MeshComponent",
    "MeshComponentExtra",
    "SpatialDatasetsComponent",
    "TablesComponent",
    "VectorComponent",
]
