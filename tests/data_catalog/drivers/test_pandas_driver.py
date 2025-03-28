import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from hydromt._compat import HAS_OPENPYXL
from hydromt.data_catalog.drivers.dataframe import PandasDriver


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
        return PandasDriver()

    # lazy-fixtures not maintained:
    # https://github.com/TvoroG/pytest-lazy-fixture/issues/65#issuecomment-1914527162
    uri_xls_param = pytest.param(
        "uri_xls",
        marks=pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl is not installed"),
    )
    uri_xlsx_param = pytest.param(
        "uri_xlsx",
        marks=pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl is not installed"),
    )
    fixture_uris = pytest.mark.parametrize(
        "uri", ["uri_csv", "uri_parquet", uri_xls_param, uri_xlsx_param, "uri_fwf"]
    )
    fixture_uris_no_fwf = pytest.mark.parametrize(
        "uri", ["uri_csv", "uri_parquet", uri_xls_param, uri_xlsx_param]
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
        uris: List[str] = [request.getfixturevalue(uri)]
        new_df: pd.DataFrame = driver.read(uris).sort_values("id")
        pd.testing.assert_frame_equal(df, new_df)

    @fixture_uris_no_fwf
    def test_read_with_filters(
        self,
        uri: str,
        request: pytest.FixtureRequest,
        driver: PandasDriver,
    ):
        uris: List[str] = [request.getfixturevalue(uri)]
        variables = ["city", "country"]
        df: pd.DataFrame = driver.read(uris, variables=variables)
        assert df.columns.to_list() == variables

    temp_xls_param = pytest.param(
        "temp.xls",
        marks=pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl is not installed"),
    )
    temp_xlsx_param = pytest.param(
        "temp.xlsx",
        marks=pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl is not installed"),
    )

    @pytest.mark.parametrize(
        "filename", ["temp.csv", "temp.parquet", temp_xls_param, temp_xlsx_param]
    )
    def test_write(
        self, filename: str, df: pd.DataFrame, tmp_dir: Path, driver: PandasDriver
    ):
        df_path = tmp_dir / filename
        driver.write(df_path, df, index=False)
        reread = driver.read([str(df_path)])
        assert np.all(reread == df)

    temp_2_xlsx_param = pytest.param(
        "temp_2.xlsx",
        marks=pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl is not installed"),
    )
    temp_2_xls_param = pytest.param(
        "temp_2.xls",
        marks=pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl is not installed"),
    )

    @pytest.mark.parametrize(
        "filename", ["temp_2.csv", temp_2_xls_param, temp_2_xlsx_param]
    )
    def test_handles_index_col(self, filename: str, df: pd.DataFrame, tmp_dir: Path):
        df_path = tmp_dir / filename
        driver = PandasDriver(options={"index_col": 0})
        driver.write(df_path, df)

        vars_slice = ["city", "country"]
        df_filtered = driver.read([str(df_path)], variables=vars_slice)
        assert np.all(df_filtered.columns == vars_slice)
