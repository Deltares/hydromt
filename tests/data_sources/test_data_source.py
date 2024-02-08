import pytest

from hydromt.data_sources import DataSource, GeoDataFrameDataSource
from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver
from hydromt.metadata_resolvers import MetaDataResolver


class TestDataSource:
    def test_submodel_validate(
        self, mock_driver: GeoDataFrameDriver, mock_resolver: MetaDataResolver
    ):
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
