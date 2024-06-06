"""Data Adapters are generic for its HydroMT type and perform transformations."""

# TODO: correct imports after deprecation of old adapters
from .data_adapter import DataAdapter
from .dataframe import DataFrameAdapter
from .dataset import DatasetAdapter
from .geodataframe import GeoDataFrameAdapter
from .geodataset import GeoDatasetAdapter
from .rasterdataset import RasterDatasetAdapter

__all__ = [
    "DataAdapter",
    "GeoDataFrameAdapter",
    "DataFrameAdapter",
    "GeoDatasetAdapter",
    "RasterDatasetAdapter",
    "DatasetAdapter",
]
