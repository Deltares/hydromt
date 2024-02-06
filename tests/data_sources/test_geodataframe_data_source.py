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
            root=".",
            name="geojsonfile",
            data_type="GeoDataFrame",
            driver=mock_driver,
            metadata_resolver=mock_resolver,
            uri="testuri",
        )

    def test_validators(self):
        with pytest.raises(ValidationError) as e_info:
            GeoDataFrameDataSource(
                root=".",
                name="name",
                data_type="GeoDataFrame",
                uri="uri",
                metadata_resolver="does not exist",
                driver="does not exist",
            )

        assert e_info.value.error_count() == 2
        error_meta = next(
            filter(lambda e: e["loc"] == ("metadata_resolver",), e_info.value.errors())
        )
        assert error_meta["type"] == "value_error"
        error_driver = next(
            filter(lambda e: e["loc"] == ("driver",), e_info.value.errors())
        )
        assert error_driver["type"] == "value_error"

    def test_model_validate(
        self, mock_driver: GeoDataFrameDriver, mock_resolver: MetaDataResolver
    ):
        GeoDataFrameDataSource.model_validate(
            {
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_driver,
                "metadata_resolver": mock_resolver,
                "uri": "test_uri",
            }
        )
        with pytest.raises(
            ValidationError, match="'data_type' must be 'GeoDataFrame'."
        ):
            GeoDataFrameDataSource.model_validate(
                {
                    "name": "geojsonfile",
                    "data_type": "DifferentDataType",
                    "driver": mock_driver,
                    "metadata_resolver": mock_resolver,
                    "uri": "test_uri",
                }
            )

    def test_read_data(
        self, geodf: gpd.GeoDataFrame, example_source: GeoDataFrameDataSource
    ):
        gdf1 = example_source.read_data(bbox=list(geodf.total_bounds))
        assert isinstance(gdf1, gpd.GeoDataFrame)
        assert np.all(gdf1 == geodf)
        example_source.rename = {"test": "test1"}
        gdf1 = example_source.read_data(bbox=list(geodf.total_bounds), buffer=1000)
        assert np.all(gdf1 == geodf)
