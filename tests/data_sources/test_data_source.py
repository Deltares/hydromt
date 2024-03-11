from typing import Any, Dict

import pytest

from hydromt.data_adapter.geodataframe import GeoDataFrameAdapter
from hydromt.data_source import DataSource, GeoDataFrameSource
from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver


class TestDataSource:
    def test_polymorphism(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
    ):
        submodel: DataSource = DataSource.model_validate(
            {
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": mock_geodataframe_adapter,
                "uri": "test_uri",
            }
        )
        assert isinstance(submodel, GeoDataFrameSource)
        with pytest.raises(ValueError, match="Unknown 'data_type'"):
            DataSource.model_validate(
                {
                    "name": "geojsonfile",
                    "data_type": "Bogus",
                    "driver": mock_geodf_driver,
                    "data_adapter": mock_geodataframe_adapter,
                    "uri": "test_uri",
                }
            )

    def test_summary(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
    ):
        class MockGeoDataFrameAdapter(GeoDataFrameAdapter):
            def transform(self, ds, **kwargs):
                return None

        mock_geodataframe_adapter = MockGeoDataFrameAdapter(
            harmonization_settings={"meta": {"custom_meta": "test"}}
        )

        submodel: DataSource = DataSource.model_validate(
            {
                "root": "/",
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": mock_geodataframe_adapter,
                "uri": "test_uri",
            }
        )
        sum: Dict[str, Any] = submodel.summary()
        assert sum["data_type"] == "GeoDataFrame"
        assert sum["uri"] == "/test_uri"
        assert sum["driver"] == "MockGeoDataFrameDriver"
        assert sum["custom_meta"] == "test"
