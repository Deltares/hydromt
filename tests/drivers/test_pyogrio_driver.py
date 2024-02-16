from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyogrio.errors import DataSourceError

from hydromt._typing import Bbox
from hydromt.drivers.pyogrio_driver import PyogrioDriver
from hydromt.region.region import Region


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

    def test_read_nodata(self):
        driver = PyogrioDriver()
        with pytest.raises(DataSourceError):
            driver.read("no_data.geojson")

    def test_read_with_filters(self, uri: str):
        driver = PyogrioDriver()
        bbox: Bbox = (-60, -34.5600, -55, -30)
        region = Region({"bbox": bbox, "buffer": 10000}).construct()
        gdf: gpd.GeoDataFrame = driver.read(uri, region=region)
        assert gdf.shape == (1, 4)
        gdf = driver.read(uri, region=region)
        assert gdf.shape == (1, 4)
