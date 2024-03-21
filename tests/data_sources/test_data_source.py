from typing import Any, Dict

import pytest

from hydromt.data_adapter.geodataframe import GeoDataFrameAdapter
from hydromt.data_source import DataSource, GeoDataFrameSource
from hydromt.data_source.data_source import get_nested_var, set_nested_var
from hydromt.driver.geodataframe_driver import GeoDataFrameDriver


class TestDataSource:
    def test_polymorphism_model_validate(
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

    def test_polymorphism_unknown_data_type(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
    ):
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
        root: str,
    ):
        class MockGeoDataFrameAdapter(GeoDataFrameAdapter):
            def transform(self, ds, **kwargs):
                return None

        mock_gdf_adapter = MockGeoDataFrameAdapter(meta={"custom_meta": "test"})

        submodel: DataSource = DataSource.model_validate(
            {
                "root": root,
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": mock_gdf_adapter,
                "uri": "test_uri",
                "meta": {"custom_meta": "test"},
            }
        )
        summ: Dict[str, Any] = submodel.summary()
        assert summ["data_type"] == "GeoDataFrame"
        assert summ["uri"] == f"{root}test_uri"
        assert summ["driver"] == "MockGeoDataFrameDriver"
        assert summ["custom_meta"] == "test"


class TestGetNestedVar:
    def test_reads_nested(
        self,
    ):
        class FakeGeoDfDriver(GeoDataFrameDriver):
            def read(self, **kwargs):
                pass

        mock_geodf_driver = FakeGeoDfDriver(metadata_resolver={"name": "convention"})

        submodel: DataSource = DataSource.model_validate(
            {
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": {"unit_add": {"var1": 1}},
                "uri": "test_uri",
            }
        )
        assert get_nested_var(["driver", "metadata_resolver", "unit_add"], submodel), {
            "var1": 1
        }


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
