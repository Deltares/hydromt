import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyogrio.errors import DataSourceError
from shapely import box

from hydromt._typing import Bbox
from hydromt.drivers.pyogrio_driver import PyogrioDriver


class TestPyogrioDriver:
    @pytest.fixture(scope="class")
    def uri_gjson(self, geodf: gpd.GeoDataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.geojson")
        geodf.to_file(uri, driver="GeoJSON")
        return uri

    @pytest.fixture(scope="class")
    def uri_shp(self, geodf: gpd.GeoDataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.shp")
        geodf.to_file(uri, "ESRI Shapefile")
        return uri

    @pytest.fixture(scope="class")
    def uri_gpkg(self, geodf: gpd.GeoDataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.gpkg")
        geodf.to_file(uri, driver="GPKG")
        return uri

    @pytest.fixture(scope="class")
    def uri_fgb(self, geodf: gpd.GeoDataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.fgb")
        geodf.to_file(uri, driver="FlatGeobuf")
        return uri

    # lazy-fixtures not maintained:
    # https://github.com/TvoroG/pytest-lazy-fixture/issues/65#issuecomment-1914527162
    fixture_uris = pytest.mark.parametrize(
        "uri",
        ["uri_gjson", "uri_shp", "uri_fgb", "uri_gpkg"],
    )

    @pytest.fixture(scope="class")
    def _raise_gdal_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            yield

    @pytest.mark.usefixtures("_raise_gdal_warnings")
    @fixture_uris
    def test_read(
        self, uri: str, request: pytest.FixtureRequest, geodf: gpd.GeoDataFrame
    ):
        uri = request.getfixturevalue(uri)
        driver = PyogrioDriver()
        gdf: gpd.GeoDataFrame = driver.read(uri).sort_values("id")
        assert np.all(gdf.reset_index(drop=True) == geodf)  # fgb scrambles order

    @pytest.mark.usefixtures("_raise_gdal_warnings")
    def test_read_nodata(self):
        driver = PyogrioDriver()
        with pytest.raises(DataSourceError):
            driver.read("no_data.geojson")

    @pytest.mark.usefixtures("_raise_gdal_warnings")
    @fixture_uris
    def test_read_with_filters(self, uri: str, request: pytest.FixtureRequest):
        uri = request.getfixturevalue(uri)
        driver = PyogrioDriver()
        bbox: Bbox = (-60, -34.5600, -55, -30)
        gdf: gpd.GeoDataFrame = driver.read(
            uri, bbox=(-60, -34.5600, -55, -30), buffer=10000
        )
        assert gdf.shape == (1, 4)
        gdf = driver.read(uri, mask=gpd.GeoSeries(box(*bbox)), buffer=10000)
        assert gdf.shape == (1, 4)
        with pytest.raises(ValueError, match="Both 'bbox' and 'mask' are provided."):
            driver.read(uri, bbox=bbox, mask=gpd.GeoSeries(box(*bbox)), buffer=10000)