import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydromt.data_source import SourceMetadata
from hydromt.drivers.pandas_driver import PandasDriver
from hydromt.metadata_resolver.convention_resolver import ConventionResolver
from hydromt.metadata_resolver.metadata_resolver import MetaDataResolver


class TestPandasDriver:
    @pytest.fixture()
    def metadata(self):
        return SourceMetadata()

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
    def uri_xlsx(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.xlsx")
        df.to_excel(uri)
        return uri

    @pytest.fixture(scope="class")
    def uri_xls(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.xls")
        df.to_excel(uri, engine="openpyxl")
        return uri

    @pytest.fixture(scope="class")
    def uri_fwf(self, df: pd.DataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.fwf")
        df.to_string(uri)
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
        metadata: SourceMetadata,
    ):
        uri = request.getfixturevalue(uri)
        new_df: pd.DataFrame = driver.read(uri, metadata).sort_values("id")
        unnamed_col_name = "Unnamed: 0"  # some ui methods struggle with this.
        if unnamed_col_name in new_df.columns:
            new_df.drop(unnamed_col_name, axis=1, inplace=True)
        pd.testing.assert_frame_equal(df, new_df)

    def test_read_nodata(self, driver: PandasDriver, metadata: SourceMetadata):
        with pytest.raises(FileNotFoundError):
            driver.read("no_data.geojson", metadata)

    def test_read_multiple_uris(self, metadata: SourceMetadata):
        # Create Resolver that returns multiple uris
        class FakeResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return ["more", "than", "one"]

        driver: PandasDriver = PandasDriver(
            metadata_resolver=FakeResolver(),
        )
        with pytest.raises(ValueError, match="not supported"):
            driver.read("uri_{variable}", metadata, variables=["more", "than", "one"])

    @fixture_uris_no_fwf
    def test_read_with_filters(
        self,
        uri: str,
        request: pytest.FixtureRequest,
        driver: PandasDriver,
        metadata: SourceMetadata,
    ):
        uri = request.getfixturevalue(uri)
        variables = ["city", "country"]
        df: pd.DataFrame = driver.read(uri, metadata, variables=variables)
        assert df.columns.to_list() == variables

    @pytest.mark.parametrize(
        "filename", ["temp.csv", "temp.parquet", "temp.xls", "temp.xlsx"]
    )
    def test_write(
        self, filename: str, df: pd.DataFrame, tmp_dir: Path, metadata: SourceMetadata
    ):
        df_path = tmp_dir / filename
        driver = PandasDriver()
        driver.write(df_path, df)
        reread = driver.read(str(df_path), metadata)
        unnamed_col_name = "Unnamed: 0"  # some ui methods struggle with this.
        if unnamed_col_name in reread.columns:
            reread.drop(unnamed_col_name, axis=1, inplace=True)

        assert np.all(reread == df)
