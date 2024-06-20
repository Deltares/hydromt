"""Drivers responsible for reading and writing data."""

from .base_driver import BaseDriver
from .dataframe import DataFrameDriver, PandasDriver
from .dataset import DatasetDriver, DatasetXarrayDriver
from .geodataframe import GeoDataFrameDriver, GeoDataFrameTableDriver, PyogrioDriver
from .geodataset import GeoDatasetDriver, GeoDatasetVectorDriver, GeoDatasetXarrayDriver
from .preprocessing import (
    harmonise_dims,
    remove_duplicates,
    round_latlon,
    to_datetimeindex,
)
from .raster.raster_dataset_driver import RasterDatasetDriver
from .raster.raster_xarray_driver import RasterDatasetXarrayDriver
from .raster.rasterio_driver import RasterioDriver

__all__ = [
    "BaseDriver",
    "DatasetDriver",
    "DatasetXarrayDriver",
    "DataFrameDriver",
    "GeoDataFrameDriver",
    "GeoDataFrameTableDriver",
    "GeoDatasetDriver",
    "GeoDatasetVectorDriver",
    "GeoDatasetXarrayDriver",
    "PandasDriver",
    "PyogrioDriver",
    "RasterDatasetDriver",
    "RasterDatasetXarrayDriver",
    "RasterioDriver",
    "harmonise_dims",
    "remove_duplicates",
    "round_latlon",
    "to_datetimeindex",
]

# define hydromt driver entry points
# see also hydromt.driver group in pyproject.toml
__hydromt_eps__ = [
    "DatasetXarrayDriver",
    "GeoDataFrameTableDriver",
    "GeoDatasetVectorDriver",
    "GeoDatasetXarrayDriver",
    "PandasDriver",
    "PyogrioDriver",
    "RasterDatasetXarrayDriver",
    "RasterioDriver",
]
