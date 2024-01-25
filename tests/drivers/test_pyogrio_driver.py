from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely import box

from hydromt.drivers.pyogrio_driver import PyogrioDriver
from hydromt.typing import Bbox


class TestPyogrioDriver:
    @pytest.fixture(scope="class")
    def uri(self, geodf: gpd.GeoDataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.geojson")
        geodf.to_file(uri, driver="GeoJSON")
        return uri

    def test_read(self, uri: str, geodf: gpd.GeoDataFrame):
        driver = PyogrioDriver()
        gdf: gpd.GeoDataFrame = driver.read(uri)
        assert np.all(gdf == geodf)

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
