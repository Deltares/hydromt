import pytest

from hydromt.data_adapter import GeoDataFrameAdapter
from hydromt.data_source import DataSource, GeoDataFrameSource, create_source
from hydromt.drivers import GeoDataFrameDriver


class TestCreateSource:
    def test_creates_correct_submodel(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
    ):
        submodel: DataSource = create_source(
            {
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": mock_geodataframe_adapter,
                "uri": "test_uri",
            }
        )
        assert isinstance(submodel, GeoDataFrameSource)

    def test_unknown_data_type(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
    ):
        with pytest.raises(ValueError, match="Unknown 'data_type'"):
            create_source(
                {
                    "name": "geojsonfile",
                    "data_type": "Bogus",
                    "driver": mock_geodf_driver,
                    "data_adapter": mock_geodataframe_adapter,
                    "uri": "test_uri",
                }
            )

    def test_no_data_type(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
    ):
        with pytest.raises(ValueError, match="needs 'data_type'"):
            create_source(
                {
                    "name": "geojsonfile",
                    "driver": mock_geodf_driver,
                    "data_adapter": mock_geodataframe_adapter,
                    "uri": "test_uri",
                }
            )
