import pytest

from hydromt.data_adapter.geodataframe import GeoDataFrameAdapter
from hydromt.data_sources import DataSource, GeoDataSource
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
        assert isinstance(submodel, GeoDataSource)
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
