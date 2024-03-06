import pytest

from hydromt.data_sources import DataSource, GeoDataSource
from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver
from hydromt.metadata_resolvers import MetaDataResolver


class TestDataSource:
    def test_polymorphism(
        self, mock_geodf_driver: GeoDataFrameDriver, mock_resolver: MetaDataResolver
    ):
        submodel: DataSource = DataSource.model_validate(
            {
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "metadata_resolver": mock_resolver,
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
                    "metadata_resolver": mock_resolver,
                    "uri": "test_uri",
                }
            )
