import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyogrio import write_dataframe
from pytest_mock import MockerFixture
from shapely import box

from hydromt._typing import Bbox
from hydromt.data_catalog.drivers import PyogrioDriver


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
        return PyogrioDriver()

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
        uris = [request.getfixturevalue(uri)]
        gdf: gpd.GeoDataFrame = driver.read(uris).sort_values("id")
        assert np.all(gdf.reset_index(drop=True) == geodf)  # fgb scrambles order

    def test_read_multiple_uris(self, driver: PyogrioDriver):
        # Create Resolver that returns multiple uris
        with pytest.raises(ValueError, match="not supported"):
            driver.read(["more", "than", "one"])

    @pytest.mark.usefixtures("_raise_gdal_warnings")
    @fixture_uris
    def test_read_with_filters(
        self,
        uri: str,
        request: pytest.FixtureRequest,
        driver: PyogrioDriver,
    ):
        uris = [request.getfixturevalue(uri)]
        bbox: Bbox = (-60, -34.5600, -55, -30)
        mask = gpd.GeoSeries(box(*bbox), crs=4326).to_crs(3857).buffer(10000)
        gdf = driver.read(uris, mask=mask)
        assert gdf.shape == (1, 4)

    @fixture_uris
    def test_read_variables(
        self,
        uri: str,
        request: pytest.FixtureRequest,
        driver: PyogrioDriver,
    ):
        uris = [request.getfixturevalue(uri)]
        variables = ["country"]
        gdf = driver.read(uris, variables=variables)
        assert set(gdf.columns) == set(variables + ["geometry"])

    def test_write(self, geodf: gpd.GeoDataFrame, tmp_dir: Path):
        df_path = tmp_dir / "temp.gpkg"
        driver = PyogrioDriver()
        driver.write(df_path, geodf)
        assert np.all(driver.read([str(df_path)]) == geodf)

    def test_write_unknown_uri(
        self, geodf: gpd.GeoDataFrame, tmp_dir: Path, mocker: MockerFixture
    ):
        df_path = tmp_dir / "temp.fakeformat"
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataframe.pyogrio_driver.write_dataframe",
            spec=write_dataframe,
        )
        driver = PyogrioDriver()
        driver.write(df_path, geodf)
        assert mock_xr_open.call_args[0][1] == str(tmp_dir / "temp.fgb")
