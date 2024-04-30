import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely import box

from hydromt._typing import Bbox
from hydromt.drivers import PyogrioDriver
from hydromt.metadata_resolver.convention_resolver import ConventionResolver
from hydromt.metadata_resolver.metadata_resolver import MetaDataResolver


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

    @pytest.fixture(scope="class")
    def driver(self):
        return PyogrioDriver(metadata_resolver=ConventionResolver())

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
        self,
        uri: str,
        request: pytest.FixtureRequest,
        geodf: gpd.GeoDataFrame,
        driver: PyogrioDriver,
    ):
        uri = request.getfixturevalue(uri)
        gdf: gpd.GeoDataFrame = driver.read(uri).sort_values("id")
        assert np.all(gdf.reset_index(drop=True) == geodf)  # fgb scrambles order

    @pytest.mark.usefixtures("_raise_gdal_warnings")
    def test_read_nodata(self, driver: PyogrioDriver):
        with pytest.raises(FileNotFoundError):
            driver.read("no_data.geojson")

    def test_read_multiple_uris(self):
        # Create Resolver that returns multiple uris
        class FakeResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return ["more", "than", "one"]

        driver: PyogrioDriver = PyogrioDriver(
            metadata_resolver=FakeResolver(),
        )
        with pytest.raises(ValueError, match="must be 1"):
            driver.read("uri_{variable}", variables=["more", "than", "one"])

    @pytest.mark.usefixtures("_raise_gdal_warnings")
    @fixture_uris
    def test_read_with_filters(
        self, uri: str, request: pytest.FixtureRequest, driver: PyogrioDriver
    ):
        uri = request.getfixturevalue(uri)
        bbox: Bbox = (-60, -34.5600, -55, -30)
        mask = gpd.GeoSeries(box(*bbox), crs=4326).to_crs(3857).buffer(10000)
        gdf = driver.read(uri, mask=mask)
        assert gdf.shape == (1, 4)

    def test_write(self, geodf: gpd.GeoDataFrame, tmp_dir: Path):
        df_path = tmp_dir / "temp.gpkg"
        driver = PyogrioDriver()
        driver.write(geodf, df_path)
        assert np.all(driver.read(str(df_path)) == geodf)
