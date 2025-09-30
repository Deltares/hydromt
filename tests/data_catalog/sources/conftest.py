import uuid
from typing import Type

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

from hydromt._typing import SourceMetadata
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


# Helper function to generate unique driver names and classes
def make_driver_class(base_cls, name_prefix, supports_writing=True, extra_attrs=None):
    extra_attrs = extra_attrs or {}
    class_name = f"{name_prefix}_{uuid.uuid4().hex[:8]}"
    driver_name = f"{name_prefix.lower()}_{uuid.uuid4().hex[:8]}"
    attrs = {"name": driver_name, "supports_writing": supports_writing, **extra_attrs}
    return type(class_name, (base_cls,), attrs)


# DataFrame
@pytest.fixture(scope="session")
def mock_df_adapter():
    class MockDataFrameAdapter(DataFrameAdapter):
        def transform(self, df: pd.DataFrame, *args, **kwargs):
            return df

    return MockDataFrameAdapter()


@pytest.fixture
def MockDataFrameDriver(df: pd.DataFrame) -> Type[DataFrameDriver]:
    class MockDataFrameDriver(DataFrameDriver):
        name = f"mock_df_driver_{uuid.uuid4().hex[:8]}"
        supports_writing = True

        def write(self, *args, **kwargs) -> None:
            pass

        def read(self, *args, **kwargs) -> pd.DataFrame:
            return df

    return MockDataFrameDriver


@pytest.fixture
def MockDataFrameReadOnlyDriver(MockDataFrameDriver) -> Type[DataFrameDriver]:
    return make_driver_class(
        MockDataFrameDriver, "MockDataFrameReadOnlyDriver", supports_writing=False
    )


# Dataset
@pytest.fixture(scope="session")
def mock_ds_adapter():
    class MockDatasetAdapter(DatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, *args, **kwargs):
            return ds

    return MockDatasetAdapter()


@pytest.fixture
def MockDatasetDriver(timeseries_ds: xr.Dataset) -> Type[DatasetDriver]:
    class MockDatasetDriver(DatasetDriver):
        name = f"mock_ds_driver_{uuid.uuid4().hex[:8]}"
        supports_writing = True

        def write(self, *args, **kwargs) -> None:
            pass

        def read(self, *args, **kwargs) -> xr.Dataset:
            return timeseries_ds

    return MockDatasetDriver


@pytest.fixture
def MockDatasetReadOnlyDriver(MockDatasetDriver) -> Type[DatasetDriver]:
    return make_driver_class(
        MockDatasetDriver, "MockDatasetReadOnlyDriver", supports_writing=False
    )


# GeoDataFrame
@pytest.fixture
def MockGeoDataFrameDriver(geodf: gpd.GeoDataFrame) -> Type[GeoDataFrameDriver]:
    class MockGeoDataFrameDriver(GeoDataFrameDriver):
        name = f"mock_geodf_driver_{uuid.uuid4().hex[:8]}"
        supports_writing = True

        def write(self, *args, **kwargs) -> None:
            pass

        def read(self, *args, **kwargs) -> gpd.GeoDataFrame:
            return geodf

    return MockGeoDataFrameDriver


@pytest.fixture
def MockGeoDataFrameReadOnlyDriver(MockGeoDataFrameDriver) -> Type[GeoDataFrameDriver]:
    return make_driver_class(
        MockGeoDataFrameDriver, "MockGeoDataFrameReadOnlyDriver", supports_writing=False
    )


# GeoDataset
@pytest.fixture
def MockGeoDatasetDriver(geoda: xr.DataArray) -> Type[GeoDatasetDriver]:
    class MockGeoDatasetDriver(GeoDatasetDriver):
        name = f"mock_geods_driver_{uuid.uuid4().hex[:8]}"
        supports_writing = True

        def write(self, *args, **kwargs) -> None:
            pass

        def read(self, *args, **kwargs) -> xr.Dataset:
            return geoda.to_dataset()

    return MockGeoDatasetDriver


@pytest.fixture
def MockGeoDatasetReadOnlyDriver(MockGeoDatasetDriver) -> Type[GeoDatasetDriver]:
    return make_driver_class(
        MockGeoDatasetDriver, "MockGeoDatasetReadOnlyDriver", supports_writing=False
    )


@pytest.fixture
def mock_geo_ds_adapter():
    class MockGeoDataSetAdapter(GeoDatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, *args, **kwargs):
            return ds

    return MockGeoDataSetAdapter()


# RasterDataset
@pytest.fixture
def MockRasterDatasetDriver(raster_ds: xr.Dataset) -> Type[RasterDatasetDriver]:
    class MockRasterDatasetDriver(RasterDatasetDriver):
        name = f"mock_raster_ds_driver_{uuid.uuid4().hex[:8]}"
        supports_writing = True

        def write(self, *args, **kwargs) -> None:
            pass

        def read(self, *args, **kwargs) -> xr.Dataset:
            return raster_ds

    return MockRasterDatasetDriver


@pytest.fixture
def MockRasterDatasetReadOnlyDriver(
    MockRasterDatasetDriver,
) -> Type[RasterDatasetDriver]:
    return make_driver_class(
        MockRasterDatasetDriver,
        "MockRasterDatasetReadOnlyDriver",
        supports_writing=False,
    )


@pytest.fixture
def mock_raster_ds_adapter():
    class MockRasterDataSetAdapter(RasterDatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, *args, **kwargs):
            return ds

    return MockRasterDataSetAdapter()
