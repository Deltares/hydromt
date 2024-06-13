from typing import List, Type

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
from hydromt.data_catalog.drivers import DataFrameDriver, DatasetDriver

# DataFrame


@pytest.fixture(scope="session")
def mock_df_adapter():
    class MockDataFrameAdapter(DataFrameAdapter):
        def transform(self, df: pd.DataFrame, metadata: SourceMetadata, **kwargs):
            return df

    return MockDataFrameAdapter()


@pytest.fixture(scope="class")
def MockDataFrameDriver(df: pd.DataFrame) -> Type[DataFrameDriver]:
    class MockDataFrameDriver(DataFrameDriver):
        name = "mock_df_driver"
        supports_writing = True

        def write(self, path: StrPath, df: pd.DataFrame, **kwargs) -> None:
            pass

        def read(self, uri: str, **kwargs) -> pd.DataFrame:
            return self.read_data([uri], **kwargs)

        def read_data(self, uris: List[str], **kwargs) -> pd.DataFrame:
            return df

    return MockDataFrameDriver


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

        def read(self, uri: str, **kwargs) -> xr.Dataset:
            return self.read_data([uri], **kwargs)

        def read_data(self, uris: List[str], **kwargs) -> xr.Dataset:
            return timeseries_ds

    return MockDatasetDriver


# GeoDataset


@pytest.fixture()
def mock_geo_ds_adapter():
    class MockGeoDataSetAdapter(GeoDatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, **kwargs):
            return ds

    return MockGeoDataSetAdapter()


# RasterDataset


@pytest.fixture()
def mock_raster_ds_adapter():
    class MockRasterDataSetAdapter(RasterDatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, **kwargs):
            return ds

    return MockRasterDataSetAdapter()
