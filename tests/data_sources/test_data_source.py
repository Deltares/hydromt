from unittest.mock import MagicMock

import geopandas as gpd
import pytest

from hydromt.data_sources import DataSource, GeoDataFrameDataSource
from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver
from hydromt.metadata_resolvers import MetaDataResolver


class TestDataSource:
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

    def test_submodel_validate(self, mock_driver, mock_resolver):
        submodel: DataSource = DataSource.submodel_validate(
            {
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_driver,
                "metadata_resolver": mock_resolver,
                "uri": "test_uri",
            }
        )
        assert isinstance(submodel, GeoDataFrameDataSource)
        with pytest.raises(ValueError, match="Unknown 'data_type'"):
            DataSource.submodel_validate(
                {
                    "name": "geojsonfile",
                    "data_type": "Bogus",
                    "driver": mock_driver,
                    "metadata_resolver": mock_resolver,
                    "uri": "test_uri",
                }
            )
