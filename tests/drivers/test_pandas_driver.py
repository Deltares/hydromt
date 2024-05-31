import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydromt.data_catalog.drivers.dataframe import PandasDriver
from hydromt.data_catalog.uri_resolvers.convention_resolver import ConventionResolver
from hydromt.data_catalog.uri_resolvers.metadata_resolver import MetaDataResolver


class TestPandasDriver:
    @pytest.fixture(scope="class")
    def uri_csv(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.csv")
        df.to_csv(uri, index=False)
        return uri

    @pytest.fixture(scope="class")
    def uri_parquet(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.parquet")
        df.to_parquet(uri, index=False)
        return uri

    @pytest.fixture(scope="class")
    def uri_xlsx(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.xlsx")
        df.to_excel(uri, index=False)
        return uri

    @pytest.fixture(scope="class")
    def uri_xls(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.xls")
        df.to_excel(uri, engine="openpyxl", index=False)
        return uri

    @pytest.fixture(scope="class")
    def uri_fwf(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.fwf")
        df.to_string(uri, index=False)
        return uri

    @pytest.fixture(scope="class")
    def driver(self):
        return PandasDriver(metadata_resolver=ConventionResolver())

    # lazy-fixtures not maintained:
    # https://github.com/TvoroG/pytest-lazy-fixture/issues/65#issuecomment-1914527162
    fixture_uris = pytest.mark.parametrize(
        "uri", ["uri_csv", "uri_parquet", "uri_xls", "uri_xlsx", "uri_fwf"]
    )
    fixture_uris_no_fwf = pytest.mark.parametrize(
        "uri", ["uri_csv", "uri_parquet", "uri_xls", "uri_xlsx"]
    )

    @pytest.fixture(scope="class")
    def _raise_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            yield

    @fixture_uris
    def test_read(
        self,
        uri: str,
        request: pytest.FixtureRequest,
        df: pd.DataFrame,
        driver: PandasDriver,
    ):
        uri = request.getfixturevalue(uri)
        new_df: pd.DataFrame = driver.read(uri).sort_values("id")
        pd.testing.assert_frame_equal(df, new_df)

    def test_read_nodata(self, driver: PandasDriver):
        with pytest.raises(FileNotFoundError):
            driver.read("no_data.geojson")

    def test_read_multiple_uris(self):
        # Create Resolver that returns multiple uris
        class FakeResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return ["more", "than", "one"]

        driver: PandasDriver = PandasDriver(
            metadata_resolver=FakeResolver(),
        )
        with pytest.raises(ValueError, match="not supported"):
            driver.read("uri_{variable}", variables=["more", "than", "one"])

    @fixture_uris_no_fwf
    def test_read_with_filters(
        self,
        uri: str,
        request: pytest.FixtureRequest,
        driver: PandasDriver,
    ):
        uri = request.getfixturevalue(uri)
        variables = ["city", "country"]
        df: pd.DataFrame = driver.read(uri, variables=variables)
        assert df.columns.to_list() == variables

    @pytest.mark.parametrize(
        "filename", ["temp.csv", "temp.parquet", "temp.xls", "temp.xlsx"]
    )
    def test_write(self, filename: str, df: pd.DataFrame, tmp_dir: Path):
        df_path = tmp_dir / filename
        driver = PandasDriver()
        driver.write(df_path, df, index=False)
        reread = driver.read(str(df_path))
        assert np.all(reread == df)

    @pytest.mark.parametrize("filename", ["temp_2.csv", "temp_2.xls", "temp_2.xlsx"])
    def test_handles_index_col(self, filename: str, df: pd.DataFrame, tmp_dir: Path):
        df_path = tmp_dir / filename
        driver = PandasDriver(options={"index_col": 0})
        driver.write(df_path, df)

        vars_slice = ["city", "country"]
        df_filtered = driver.read(str(df_path), variables=vars_slice)
        assert np.all(df_filtered.columns == vars_slice)
