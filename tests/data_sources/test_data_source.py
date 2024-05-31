from typing import Any, Dict

import pytest

from hydromt.data_catalog.adapters.geodataframe import GeoDataFrameAdapter
from hydromt.data_catalog.drivers import GeoDataFrameDriver
from hydromt.data_catalog.sources import DataSource, GeoDataFrameSource, create_source
from hydromt.data_catalog.sources.data_source import get_nested_var, set_nested_var


class TestDataSource:
    def test_summary(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
        root: str,
    ):
        submodel: DataSource = GeoDataFrameSource.model_validate(
            {
                "root": root,
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": mock_geodataframe_adapter,
                "uri": "test_uri",
                "metadata": {"url": "www.example.com"},
            }
        )
        summ: Dict[str, Any] = submodel.summary()
        assert summ["data_type"] == "GeoDataFrame"
        assert summ["uri"] == "test_uri"
        assert summ["driver"] == "MockGeoDataFrameDriver"
        assert summ["url"] == "www.example.com"

    def test_serializes_data_type(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
        root: str,
    ):
        submodel: DataSource = GeoDataFrameSource.model_validate(
            {
                "root": root,
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": mock_geodataframe_adapter,
                "uri": "test_uri",
                "metadata": {"url": "www.example.com"},
            }
        )
        assert submodel.model_dump()["data_type"] == "GeoDataFrame"


class TestGetNestedVar:
    def test_reads_nested(
        self,
    ):
        class FakeGeoDfDriver(GeoDataFrameDriver):
            name = "test_reads_nested"

            def read_data(self, **kwargs):
                pass

        mock_geodf_driver = FakeGeoDfDriver(metadata_resolver={"name": "convention"})

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
