from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyogrio.errors import DataSourceError

from hydromt.data_sources.geodataframe_data_source import GeoDataFrameDataSource


def test_geodataframe(geodf: gpd.GeoDataFrame, tmpdir: Path):
    fn_gdf = str(tmpdir.join("test.geojson"))
    geodf.to_file(fn_gdf, driver="GeoJSON")
    gdf_source = GeoDataFrameDataSource(
        name="geojsonfile",
        driver="pyogrio",
        metadata_resolver="convention_resolver",
        uri=fn_gdf,
    )
    gdf1 = gdf_source.read_data(bbox=list(geodf.total_bounds))
    # gdf1 = data_catalog.get_geodataframe(fn_gdf, bbox=geodf.total_bounds)
    assert isinstance(gdf1, gpd.GeoDataFrame)
    assert np.all(gdf1 == geodf)
    gdf_source.rename = {"test": "test1"}
    gdf1 = gdf_source.read_data(bbox=list(geodf.total_bounds), buffer=1000)
    # gdf1 = data_catalog.get_geodataframe(
    #     "test.geojson", bbox=geodf.total_bounds, buffer=1000, rename={"test": "test1"}
    # )
    assert np.all(gdf1 == geodf)
    gdf_source.uri = "no_file.geojson"
    with pytest.raises(DataSourceError, match="No such file"):
        gdf_source.read_data()
        # data_catalog.get_geodataframe("no_file.geojson")
