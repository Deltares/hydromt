"""Drivers responsible for reading and writing data."""

from hydromt.data_catalog.drivers.base_driver import BaseDriver, DriverOptions
from hydromt.data_catalog.drivers.dataframe import DataFrameDriver, PandasDriver
from hydromt.data_catalog.drivers.dataset import DatasetDriver, DatasetXarrayDriver
from hydromt.data_catalog.drivers.geodataframe import (
    GeoDataFrameDriver,
    GeoDataFrameTableDriver,
    PyogrioDriver,
)
from hydromt.data_catalog.drivers.geodataframe.table_driver import (
    GeoDataFrameTableOptions,
)
from hydromt.data_catalog.drivers.geodataset import (
    GeoDatasetDriver,
    GeoDatasetVectorDriver,
    GeoDatasetXarrayDriver,
)
from hydromt.data_catalog.drivers.preprocessing import (
    harmonise_dims,
    remove_duplicates,
    round_latlon,
    to_datetimeindex,
)
from hydromt.data_catalog.drivers.raster.raster_dataset_driver import (
    RasterDatasetDriver,
)
from hydromt.data_catalog.drivers.raster.raster_xarray_driver import (
    RasterDatasetXarrayDriver,
)
from hydromt.data_catalog.drivers.raster.rasterio_driver import (
    RasterioDriver,
    RasterioOptions,
)
from hydromt.data_catalog.drivers.xarray_options import XarrayDriverOptions

__all__ = [
    "BaseDriver",
    "GeoDataFrameTableOptions",
    "DriverOptions",
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
    "RasterioOptions",
    "harmonise_dims",
    "remove_duplicates",
    "round_latlon",
    "to_datetimeindex",
    "XarrayDriverOptions",
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
