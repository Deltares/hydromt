from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from hydromt.data_catalog.adapters import DataFrameAdapter, GeoDataFrameAdapter
from hydromt.data_catalog.drivers import DataFrameDriver, GeoDataFrameDriver
from hydromt.data_catalog.sources import DataFrameSource, DataSource, GeoDataFrameSource
from hydromt.data_catalog.uri_resolvers import URIResolver
from hydromt.error import NoDataException


class TestDataSource:
    def test_summary(
        self,
        MockGeoDataFrameDriver: type[GeoDataFrameDriver],
        mock_gdf_adapter: GeoDataFrameAdapter,
        mock_resolver: URIResolver,
        root: str,
    ):
        driver = MockGeoDataFrameDriver()
        submodel: DataSource = GeoDataFrameSource.model_validate(
            {
                "root": root,
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": driver,
                "data_adapter": mock_gdf_adapter,
                "uri_resolver": mock_resolver,
                "uri": "test_uri",
                "metadata": {"url": "www.example.com"},
            }
        )
        summ: Dict[str, Any] = submodel.summary()
        assert summ["data_type"] == "GeoDataFrame"
        assert summ["uri"] == "test_uri"
        assert summ["driver"] == driver.__repr_name__()
        assert summ["url"] == "www.example.com"

    def test_copies_fs(
        self,
        MockGeoDataFrameDriver: type[GeoDataFrameDriver],
        mock_resolver: URIResolver,
        mock_gdf_adapter: GeoDataFrameAdapter,
        root: str,
    ):
        GeoDataFrameSource.model_validate(
            {
                "root": root,
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": MockGeoDataFrameDriver(),
                "data_adapter": mock_gdf_adapter,
                "uri_resolver": mock_resolver,
                "uri": "test_uri",
                "metadata": {"url": "www.example.com"},
            }
        )

    def test_serializes_data_type(
        self,
        MockGeoDataFrameDriver: type[GeoDataFrameDriver],
        mock_gdf_adapter: GeoDataFrameAdapter,
        mock_resolver: URIResolver,
        root: str,
    ):
        submodel: DataSource = GeoDataFrameSource.model_validate(
            {
                "root": root,
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": MockGeoDataFrameDriver(),
                "data_adapter": mock_gdf_adapter,
                "uri_resolver": mock_resolver,
                "uri": "test_uri",
                "metadata": {"url": "www.example.com"},
            }
        )
        assert submodel.model_dump()["data_type"] == "GeoDataFrame"

    def test_read_no_file_found(
        self,
        MockGeoDataFrameDriver,
        mock_gdf_adapter,
    ):
        class MockRaisingResolver(URIResolver):
            name = "raises"

            def resolve(self, uris: List[str], **kwargs):
                raise NoDataException()

        source = GeoDataFrameSource(
            name="raises",
            data_adapter=mock_gdf_adapter,
            driver=MockGeoDataFrameDriver(),
            uri_resolver=MockRaisingResolver(),
            uri="myfile",
        )
        with pytest.raises(NoDataException):
            source.read_data()

    def test_infer_default_driver(
        self,
        MockDataFrameDriver: type[DataFrameDriver],
        mock_resolver: URIResolver,
        mock_df_adapter: DataFrameAdapter,
        df: pd.DataFrame,
        managed_tmp_path: Path,
    ):
        managed_tmp_path.touch("test.xls")
        source = DataFrameSource(
            root=".",
            name="example_source",
            driver=MockDataFrameDriver(),
            uri_resolver=mock_resolver,
            data_adapter=mock_df_adapter,
            uri=str(managed_tmp_path / "test.xls"),
        )
        assert source._infer_default_driver() == DataFrameSource._fallback_driver_read
