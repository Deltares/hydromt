from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
from pydantic import ValidationError

from hydromt.data_sources.geodataframe_data_source import GeoDataFrameDataSource
from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver
from hydromt.metadata_resolvers.metadata_resolver import MetaDataResolver


class TestGeoDataFrame:
    @pytest.fixture()
    def mock_driver(self, geodf: gpd.GeoDataFrame) -> GeoDataFrameDriver:
        driver = MagicMock(spec=GeoDataFrameDriver)
        driver.read.return_value = geodf
        return driver

    @pytest.fixture()
    def mock_resolver(self) -> MetaDataResolver:
        resolver = MagicMock(spec=MetaDataResolver)

        def fake_resolve(uri: str, **kwags) -> str:
            return [uri]

        resolver.resolve = fake_resolve
        return resolver

    @pytest.fixture()
    def example_source(
        self, mock_driver: GeoDataFrameDriver, mock_resolver: MetaDataResolver
    ) -> GeoDataFrameDataSource:
        return GeoDataFrameDataSource(
            name="geojsonfile",
            data_type="GeoDataFrame",
            driver=mock_driver,
            metadata_resolver=mock_resolver,
            uri="testuri",
        )

    def test_validators(self):
        with pytest.raises(ValidationError) as e_info:
            GeoDataFrameDataSource(
                name="name",
                data_type="GeoDataFrame",
                uri="uri",
                metadata_resolver="does not exist",
                driver="does not exist",
            )

        assert e_info.value.error_count() == 2
        error0 = e_info.value.errors()[0]
        assert error0["type"] == "value_error"
        assert error0["loc"][0] == "metadata_resolver"
        error1 = e_info.value.errors()[1]
        assert error1["type"] == "value_error"
        assert error1["loc"][0] == "driver"

    def test_read_data(
        self, geodf: gpd.GeoDataFrame, example_source: GeoDataFrameDataSource
    ):
        gdf1 = example_source.read_data(bbox=list(geodf.total_bounds))
        assert isinstance(gdf1, gpd.GeoDataFrame)
        assert np.all(gdf1 == geodf)
        example_source.rename = {"test": "test1"}
        gdf1 = example_source.read_data(bbox=list(geodf.total_bounds), buffer=1000)
        assert np.all(gdf1 == geodf)
