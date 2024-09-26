from typing import Any, Dict, List

import pytest

from hydromt._typing import NoDataException
from hydromt.data_catalog.adapters.geodataframe import GeoDataFrameAdapter
from hydromt.data_catalog.drivers import GeoDataFrameDriver
from hydromt.data_catalog.sources import DataSource, GeoDataFrameSource
from hydromt.data_catalog.uri_resolvers import URIResolver


class TestDataSource:
    def test_summary(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
        mock_resolver: URIResolver,
        root: str,
    ):
        submodel: DataSource = GeoDataFrameSource.model_validate(
            {
                "root": root,
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": mock_geodataframe_adapter,
                "uri_resolver": mock_resolver,
                "uri": "test_uri",
                "metadata": {"url": "www.example.com"},
            }
        )
        summ: Dict[str, Any] = submodel.summary()
        assert summ["data_type"] == "GeoDataFrame"
        assert summ["uri"] == "test_uri"
        assert summ["driver"] == "MockGeoDataFrameDriver"
        assert summ["url"] == "www.example.com"

    def test_copies_fs(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_resolver: URIResolver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
        root: str,
    ):
        GeoDataFrameSource.model_validate(
            {
                "root": root,
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": mock_geodataframe_adapter,
                "uri_resolver": mock_resolver,
                "uri": "test_uri",
                "metadata": {"url": "www.example.com"},
            }
        )

    def test_serializes_data_type(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
        mock_resolver: URIResolver,
        root: str,
    ):
        submodel: DataSource = GeoDataFrameSource.model_validate(
            {
                "root": root,
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": mock_geodataframe_adapter,
                "uri_resolver": mock_resolver,
                "uri": "test_uri",
                "metadata": {"url": "www.example.com"},
            }
        )
        assert submodel.model_dump()["data_type"] == "GeoDataFrame"

    def test_read_no_file_found(
        self,
        mock_geodf_driver,
        mock_geodataframe_adapter,
    ):
        class MockRaisingResolver(URIResolver):
            def resolve(self, uris: List[str], **kwargs):
                raise NoDataException()

        source = GeoDataFrameSource(
            name="raises",
            data_adapter=mock_geodataframe_adapter,
            driver=mock_geodf_driver,
            uri_resolver=MockRaisingResolver(),
            uri="myfile",
        )
        with pytest.raises(NoDataException):
            source.read_data()
