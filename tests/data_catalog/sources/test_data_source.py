from typing import Any, Dict, List

import pytest

from hydromt._typing import NoDataException
from hydromt.data_catalog.adapters.geodataframe import GeoDataFrameAdapter
from hydromt.data_catalog.drivers import GeoDataFrameDriver
from hydromt.data_catalog.sources import DataSource, GeoDataFrameSource, create_source
from hydromt.data_catalog.sources.data_source import get_nested_var, set_nested_var
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


class TestGetNestedVar:
    def test_reads_nested(self, mock_geodf_driver: GeoDataFrameDriver):
        class FakeGeoDfDriver(GeoDataFrameDriver):
            name = "test_reads_nested"

            def read(self, **kwargs):
                pass

        submodel: DataSource = create_source(
            {
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": {"unit_add": {"var1": 1}},
                "uri": "test_uri",
            }
        )
        assert get_nested_var(["data_adapter", "unit_add"], submodel), {"var1": 1}


class TestSetNestedVar:
    @pytest.fixture()
    def data_source_dict(self) -> Dict[str, Any]:
        return {
            "name": "geojsonfile",
            "data_type": "GeoDataFrame",
            "driver": "zarr",
            "data_adapter": {"name": "raster", "unit_add": {"var1": 1}},
            "uri": "test_uri",
        }

    def test_sets_nested(self, data_source_dict: Dict[str, Any]):
        set_nested_var(["data_adapter", "unit_add", "var1"], data_source_dict, 2)
        assert data_source_dict["data_adapter"]["unit_add"]["var1"] == 2

    def test_changes_flat_value(self, data_source_dict: Dict[str, Any]):
        set_nested_var(["driver"], data_source_dict, "pyogrio")
        assert data_source_dict["driver"] == "pyogrio"

    def test_ignores_incompatible_field(self, data_source_dict: Dict[str, Any]):
        with pytest.raises(ValueError, match="Cannot set"):
            set_nested_var(["driver", "name"], data_source_dict, "pyogrio")

    def test_adds_missing_field(self, data_source_dict: Dict[str, Any]):
        set_nested_var(["data_adapter", "unit_mult", "var1"], data_source_dict, 2.0)
        assert data_source_dict["data_adapter"]["unit_mult"] == {"var1": 2.0}
