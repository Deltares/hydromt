import warnings
from itertools import repeat
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyogrio.errors import DataSourceError
from pytest_lazyfixture import lazy_fixture
from shapely import box

from hydromt._typing import Bbox
from hydromt.drivers.pyogrio_driver import (
    PyogrioDriver,
    PyogrioExtension,
)


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
    def _raise_gdal_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            yield

    @pytest.mark.usefixtures("_raise_gdal_warnings")
    @pytest.mark.parametrize(
        "uri",
        [lazy_fixture("uri_gjson"), lazy_fixture("uri_shp")],
    )
    def test_read(self, uri: str, geodf: gpd.GeoDataFrame):
        driver = PyogrioDriver()
        gdf: gpd.GeoDataFrame = driver.read(uri)
        assert np.all(gdf == geodf)

    @pytest.mark.usefixtures("_raise_gdal_warnings")
    def test_read_nodata(self):
        driver = PyogrioDriver()
        with pytest.raises(DataSourceError):
            driver.read("no_data.geojson")

    @pytest.mark.usefixtures("_raise_gdal_warnings")
    @pytest.mark.parametrize(
        "uri",
        [lazy_fixture("uri_gjson"), lazy_fixture("uri_shp")],
    )
    def test_read_with_filters(self, uri: str):
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


@pytest.mark.parametrize(
    ("uri", "expected"),
    zip(
        ("/posix/path.geojson", "C:\windows\path.geojson", "s3://s3/path.geojson"),
        repeat(".geojson"),
    ),
)
def test_pyogrio_extension(uri: str, expected: str):
    assert PyogrioExtension(Path(uri).suffix).value == expected
