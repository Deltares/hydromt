import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from hydromt.data_catalog.drivers.geodataframe.table_driver import (
    GeoDataFrameTableDriver,
)


class TestGeoDataFrameTableDriver:
    @pytest.fixture(scope="class")
    def uri_csv(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.csv")
        df.to_csv(uri)
        return uri

    @pytest.fixture(scope="class")
    def uri_parquet(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.parquet")
        df.to_parquet(uri)
        return uri

    @pytest.fixture(scope="class")
    def uri_xls(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.xls")
        df.to_excel(uri, engine="openpyxl")
        return uri

    @pytest.fixture(scope="class")
    def uri_xlsx(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.xlsx")
        df.to_excel(uri, engine="openpyxl")
        return uri

    # lazy-fixtures not maintained:
    # https://github.com/TvoroG/pytest-lazy-fixture/issues/65#issuecomment-1914527162
    fixture_uris = pytest.mark.parametrize(
        "uri",
        ["uri_csv", "uri_parquet", "uri_xls", "uri_xlsx"],
    )

    @pytest.fixture(scope="class")
    def _raise_gdal_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            yield

    @fixture_uris
    def test_reads_correctly(
        self, uri: str, request: pytest.FixtureRequest, geodf: gpd.GeoDataFrame
    ):
        uri = request.getfixturevalue(uri)
        driver = GeoDataFrameTableDriver()
        gdf = driver.read(uris=[uri])
        pd.testing.assert_frame_equal(gdf, geodf)

    def test_unknown_extension(self):
        driver = GeoDataFrameTableDriver()
        with pytest.raises(ValueError, match="not compatible"):
            driver.read(uris=["weird_ext.zzz"])

    def test_header_case_insensitive(
        self, tmp_dir: Path, df: pd.DataFrame, geodf: gpd.GeoDataFrame
    ):
        uri = str(tmp_dir / "test.csv")
        df = df.rename({"longitude": "LONGITUDE"})
        df.to_csv(uri)
        driver = GeoDataFrameTableDriver()
        gdf = driver.read(uris=[uri])
        pd.testing.assert_frame_equal(gdf, geodf)
