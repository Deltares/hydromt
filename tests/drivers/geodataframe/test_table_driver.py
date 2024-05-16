import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from hydromt.drivers.geodataframe.table_driver import GeoDataFrameTableDriver
from hydromt.metadata_resolver.metadata_resolver import MetaDataResolver


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
        gdf = driver.read_data(uris=[uri])
        pd.testing.assert_frame_equal(gdf, geodf)

    def test_unknown_extension(self):
        driver = GeoDataFrameTableDriver()
        with pytest.raises(ValueError, match="not compatible"):
            driver.read_data(uris=["weird_ext.zzz"])

    def test_header_case_insensitive(
        self, tmp_dir: Path, df: pd.DataFrame, geodf: gpd.GeoDataFrame
    ):
        uri = str(tmp_dir / "test.csv")
        df = df.rename({"longitude": "LONGITUDE"})
        df.to_csv(uri)
        driver = GeoDataFrameTableDriver()
        gdf = driver.read_data(uris=[uri])
        pd.testing.assert_frame_equal(gdf, geodf)

    def test_read_multiple_uris(self):
        # Create Resolver that returns multiple uris
        class FakeResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return ["more", "than", "one"]

        driver: GeoDataFrameTableDriver = GeoDataFrameTableDriver(
            metadata_resolver=FakeResolver(),
        )
        with pytest.raises(ValueError, match="not supported"):
            driver.read("uri_{variable}", variables=["more", "than", "one"])
