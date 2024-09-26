from typing import List, Type, Union

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

from hydromt._typing import SourceMetadata, StrPath
from hydromt.data_catalog.adapters import (
    DataFrameAdapter,
    DatasetAdapter,
    GeoDatasetAdapter,
    RasterDatasetAdapter,
)
from hydromt.data_catalog.drivers import (
    DataFrameDriver,
    DatasetDriver,
    GeoDataFrameDriver,
    GeoDatasetDriver,
    RasterDatasetDriver,
)

# DataFrame


@pytest.fixture(scope="session")
def mock_df_adapter():
    class MockDataFrameAdapter(DataFrameAdapter):
        def transform(self, df: pd.DataFrame, metadata: SourceMetadata, **kwargs):
            return df

    return MockDataFrameAdapter()


@pytest.fixture()
def MockDataFrameDriver(df: pd.DataFrame) -> Type[DataFrameDriver]:
    class MockDataFrameDriver(DataFrameDriver):
        name = "mock_df_driver"
        supports_writing = True

        def write(self, path: StrPath, df: pd.DataFrame, **kwargs) -> None:
            pass

        def read(self, uris: List[str], **kwargs) -> pd.DataFrame:
            return df

    return MockDataFrameDriver


@pytest.fixture()
def MockDataFrameReadOnlyDriver(MockDataFrameDriver) -> Type[DataFrameDriver]:
    class MockDataFrameReadOnlyDriver(MockDataFrameDriver):
        name = "mock_df_readonly_driver"
        supports_writing = False

    return MockDataFrameReadOnlyDriver


# Dataset


@pytest.fixture(scope="session")
def mock_ds_adapter():
    class MockDatasetAdapter(DatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, **kwargs):
            return ds

    return MockDatasetAdapter()


@pytest.fixture()
def MockDatasetDriver(timeseries_ds: xr.Dataset) -> Type[DatasetDriver]:
    class MockDatasetDriver(DatasetDriver):
        name = "mock_ds_driver"
        supports_writing = True

        def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
            pass

        def read(self, uris: List[str], **kwargs) -> xr.Dataset:
            return timeseries_ds

    return MockDatasetDriver


@pytest.fixture()
def MockDatasetReadOnlyDriver(MockDatasetDriver) -> Type[DatasetDriver]:
    class MockDatasetReadOnlyDriver(MockDatasetDriver):
        name = "mock_ds_readonly_driver"
        supports_writing = False

    return MockDatasetReadOnlyDriver


# GeoDataFrame


@pytest.fixture()
def MockGeoDataFrameDriver(geodf: gpd.GeoDataFrame) -> Type[GeoDataFrameDriver]:
    class MockGeoDataFrameDriver(GeoDataFrameDriver):
        name = "mock_geodf_driver"
        supports_writing = True

        def write(self, path: StrPath, gdf: gpd.GeoDataFrame, **kwargs) -> None:
            pass

        def read(self, *args, **kwargs) -> gpd.GeoDataFrame:
            return geodf

    return MockGeoDataFrameDriver


@pytest.fixture()
def MockGeoDataFrameReadOnlyDriver(MockGeoDataFrameDriver) -> Type[GeoDataFrameDriver]:
    class MockGeoDataFrameReadOnlyDriver(MockGeoDataFrameDriver):
        name = "mock_geodf_readonly_driver"
        supports_writing = False

    return MockGeoDataFrameReadOnlyDriver


# GeoDataset


@pytest.fixture()
def MockGeoDatasetDriver(geoda: xr.DataArray) -> Type[GeoDatasetDriver]:
    class MockGeoDatasetDriver(GeoDatasetDriver):
        name = "mock_geods_driver"
        supports_writing = True

        def write(
            self, uri: str, ds: Union[xr.DataArray, xr.Dataset], **kwargs
        ) -> None:
            pass

        def read(self, uris: List[str], **kwargs) -> xr.Dataset:
            return geoda.to_dataset()

    return MockGeoDatasetDriver


@pytest.fixture()
def MockGeoDatasetReadOnlyDriver(MockGeoDatasetDriver) -> Type[GeoDatasetDriver]:
    class MockGeoDatasetReadOnlyDriver(MockGeoDatasetDriver):
        name = "mock_geods_readonly_driver"
        supports_writing = False

    return MockGeoDatasetReadOnlyDriver


@pytest.fixture()
def mock_geo_ds_adapter():
    class MockGeoDataSetAdapter(GeoDatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, **kwargs):
            return ds

    return MockGeoDataSetAdapter()


# RasterDataset


@pytest.fixture()
def MockRasterDatasetDriver(raster_ds: xr.Dataset) -> Type[RasterDatasetDriver]:
    class MockRasterDatasetDriver(RasterDatasetDriver):
        name = "mock_raster_ds_driver"
        supports_writing = True

        def write(
            self, path: StrPath, ds: Union[xr.DataArray, xr.Dataset], **kwargs
        ) -> None:
            pass

        def read(self, uris: List[str], **kwargs) -> xr.Dataset:
            return raster_ds

    return MockRasterDatasetDriver


@pytest.fixture()
def MockRasterDatasetReadOnlyDriver(
    MockRasterDatasetDriver,
) -> Type[RasterDatasetDriver]:
    class MockRasterDatasetReadOnlyDriver(MockRasterDatasetDriver):
        name = "raster_ds_readonly_driver"
        supports_writing = False

    return MockRasterDatasetReadOnlyDriver


@pytest.fixture()
def mock_raster_ds_adapter():
    class MockRasterDataSetAdapter(RasterDatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, **kwargs):
            return ds

    return MockRasterDataSetAdapter()
