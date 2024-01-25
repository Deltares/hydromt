from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyogrio.errors import DataSourceError

from hydromt.data_sources.geodataframe_data_source import GeoDataFrameDataSource


class TestGeoDataFrame:
    @pytest.fixture(scope="class")
    def example_source(
        self, geodf: gpd.GeoDataFrame, tmp_dir: Path
    ) -> GeoDataFrameDataSource:
        fn_gdf = str(tmp_dir / "test.geojson")
        geodf.to_file(fn_gdf, driver="GeoJSON")
        return GeoDataFrameDataSource(
            name="geojsonfile",
            driver="pyogrio",
            metadata_resolver="convention_resolver",
            uri=fn_gdf,
        )

    def test_read_data(
        self, geodf: gpd.GeoDataFrame, example_source: GeoDataFrameDataSource
    ):
        gdf1 = example_source.read_data(bbox=list(geodf.total_bounds))
        assert isinstance(gdf1, gpd.GeoDataFrame)
        assert np.all(gdf1 == geodf)
        example_source.rename = {"test": "test1"}
        gdf1 = example_source.read_data(bbox=list(geodf.total_bounds), buffer=1000)
        assert np.all(gdf1 == geodf)
        example_source.uri = "no_file.geojson"
        with pytest.raises(DataSourceError, match="No such file"):
            example_source.read_data()
