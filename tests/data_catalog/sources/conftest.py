import uuid
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

from hydromt.data_catalog.adapters import (
    DataFrameAdapter,
    DatasetAdapter,
    GeoDatasetAdapter,
    RasterDatasetAdapter,
)
from hydromt.data_catalog.adapters.geodataframe import GeoDataFrameAdapter
from hydromt.data_catalog.drivers import (
    DataFrameDriver,
    DatasetDriver,
    GeoDataFrameDriver,
    GeoDatasetDriver,
    RasterDatasetDriver,
)
from hydromt.data_catalog.uri_resolvers.uri_resolver import URIResolver
from hydromt.typing import SourceMetadata

# ======================================================================================
# Helper utilities
# ======================================================================================


def make_driver_class(
    base_cls: type,
    name_prefix: str,
    *,
    supports_writing: bool = True,
    extra_attrs: dict[str, Any] | None = None,
) -> type:
    """Generate a unique mock driver class inheriting from base_cls."""
    extra_attrs = extra_attrs or {}
    hex = uuid.uuid4().hex[:8]
    class_name = f"{name_prefix}_{hex}"
    driver_name = f"{name_prefix.lower()}_{hex}"
    attrs = {"name": driver_name, "supports_writing": supports_writing, **extra_attrs}
    return type(class_name, (base_cls,), attrs)


# ======================================================================================
# Mock Adapters
# ======================================================================================


@pytest.fixture(scope="session")
def mock_df_adapter():
    class MockDataFrameAdapter(DataFrameAdapter):
        def transform(self, df: pd.DataFrame, *args, **kwargs):
            return df

    return MockDataFrameAdapter()


@pytest.fixture(scope="session")
def mock_gdf_adapter():
    class MockGeoDataFrameAdapter(GeoDataFrameAdapter):
        def transform(self, gdf: gpd.GeoDataFrame, *args, **kwargs):
            return gdf

    return MockGeoDataFrameAdapter()


@pytest.fixture(scope="session")
def mock_ds_adapter():
    class MockDatasetAdapter(DatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, *args, **kwargs):
            return ds

    return MockDatasetAdapter()


@pytest.fixture(scope="session")
def mock_geo_ds_adapter():
    class MockGeoDatasetAdapter(GeoDatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, *args, **kwargs):
            return ds

    return MockGeoDatasetAdapter()


@pytest.fixture(scope="session")
def mock_raster_ds_adapter():
    class MockRasterDatasetAdapter(RasterDatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, *args, **kwargs):
            return ds

    return MockRasterDatasetAdapter()


# ======================================================================================
# Mock Drivers
# ======================================================================================


# ---- DataFrame Driver ----
@pytest.fixture
def MockDataFrameDriver(df: pd.DataFrame, tmp_path: Path) -> type[DataFrameDriver]:
    def read(self, *args, **kwargs):
        return df

    def write(self, data, *args, **kwargs):
        self._written_data = data
        self._write_called = True
        return tmp_path

    return make_driver_class(
        DataFrameDriver,
        "MockDataFrameDriver",
        extra_attrs={"read": read, "write": write},
    )


@pytest.fixture
def MockDataFrameReadOnlyDriver(MockDataFrameDriver) -> type[DataFrameDriver]:
    return make_driver_class(
        MockDataFrameDriver,
        "MockDataFrameReadOnlyDriver",
        supports_writing=False,
    )


# ---- Dataset Driver ----
@pytest.fixture
def MockDatasetDriver(timeseries_ds: xr.Dataset, tmp_path: Path) -> type[DatasetDriver]:
    def read(self, *args, **kwargs):
        return timeseries_ds

    def write(self, data, *args, **kwargs):
        self._written_ds = data
        self._write_called = True
        return tmp_path

    return make_driver_class(
        DatasetDriver,
        "MockDatasetDriver",
        extra_attrs={"read": read, "write": write},
    )


@pytest.fixture
def MockDatasetReadOnlyDriver(MockDatasetDriver) -> type[DatasetDriver]:
    return make_driver_class(
        MockDatasetDriver,
        "MockDatasetReadOnlyDriver",
        supports_writing=False,
    )


# ---- GeoDataFrame Driver ----
@pytest.fixture
def MockGeoDataFrameDriver(
    geodf: gpd.GeoDataFrame, tmp_path: Path
) -> type[GeoDataFrameDriver]:
    def read(self, *args, **kwargs):
        return geodf

    def write(self, data, *args, **kwargs):
        self._written_gdf = data
        self._write_called = True
        return tmp_path

    return make_driver_class(
        GeoDataFrameDriver,
        "MockGeoDataFrameDriver",
        extra_attrs={"read": read, "write": write},
    )


@pytest.fixture
def MockGeoDataFrameReadOnlyDriver(MockGeoDataFrameDriver) -> type[GeoDataFrameDriver]:
    return make_driver_class(
        MockGeoDataFrameDriver,
        "MockGeoDataFrameReadOnlyDriver",
        supports_writing=False,
    )


# ---- GeoDataset Driver ----
@pytest.fixture
def MockGeoDatasetDriver(geoda: xr.DataArray, tmp_path: Path) -> type[GeoDatasetDriver]:
    def read(self, *args, **kwargs):
        return geoda.to_dataset()

    def write(self, data, *args, **kwargs):
        self._written_ds = data
        self._write_called = True
        return tmp_path

    return make_driver_class(
        GeoDatasetDriver,
        "MockGeoDatasetDriver",
        extra_attrs={"read": read, "write": write},
    )


@pytest.fixture
def MockGeoDatasetReadOnlyDriver(MockGeoDatasetDriver) -> type[GeoDatasetDriver]:
    return make_driver_class(
        MockGeoDatasetDriver,
        "MockGeoDatasetReadOnlyDriver",
        supports_writing=False,
    )


# ---- RasterDataset Driver ----
@pytest.fixture
def MockRasterDatasetDriver(
    raster_ds: xr.Dataset, tmp_path: Path
) -> type[RasterDatasetDriver]:
    def read(self, *args, **kwargs):
        return raster_ds

    def write(self, data, *args, **kwargs):
        self._written_raster = data
        self._write_called = True
        return tmp_path

    return make_driver_class(
        RasterDatasetDriver,
        "MockRasterDatasetDriver",
        extra_attrs={"read": read, "write": write},
    )


@pytest.fixture
def MockRasterDatasetReadOnlyDriver(
    MockRasterDatasetDriver,
) -> type[RasterDatasetDriver]:
    return make_driver_class(
        MockRasterDatasetDriver,
        "MockRasterDatasetReadOnlyDriver",
        supports_writing=False,
    )


# ---- Uri Resolver ----
@pytest.fixture
def mock_resolver() -> URIResolver:
    class MockURIResolver(URIResolver):
        name = "mock_resolver"

        def resolve(self, uri, *args, **kwargs):
            return [uri]

    resolver = MockURIResolver()
    return resolver
