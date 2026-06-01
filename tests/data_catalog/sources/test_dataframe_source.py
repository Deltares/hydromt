from pathlib import Path
from typing import Type

import pandas as pd
import pytest

from hydromt.data_catalog.adapters import DataFrameAdapter
from hydromt.data_catalog.drivers import DataFrameDriver
from hydromt.data_catalog.sources import DataFrameSource
from hydromt.data_catalog.uri_resolvers import URIResolver


class TestDataFrameSource:
    @pytest.fixture
    def MockDataFrameSource(
        self,
        MockDataFrameDriver: Type[DataFrameDriver],
        mock_resolver: URIResolver,
        mock_df_adapter: DataFrameAdapter,
        managed_tmp_path: Path,
    ) -> DataFrameSource:
        managed_tmp_path.touch("test.xls")
        source = DataFrameSource(
            root=".",
            name="example_source",
            driver=MockDataFrameDriver(),
            uri_resolver=mock_resolver,
            data_adapter=mock_df_adapter,
            uri=str(managed_tmp_path / "test.xls"),
        )
        return source

    def test_read_data(
        self,
        MockDataFrameSource: DataFrameSource,
        df: pd.DataFrame,
    ):
        pd.testing.assert_frame_equal(df, MockDataFrameSource.read_data())

    def test_to_file_nodata(
        self, MockDataFrameSource: DataFrameSource, managed_tmp_path: Path, mocker
    ):
        output_path = managed_tmp_path / "output.csv"
        mocker.patch(
            "hydromt.data_catalog.sources.dataframe.DataFrameSource.read_data",
            return_value=None,
        )
        p = MockDataFrameSource.to_file(output_path)
        assert p is None

    @pytest.mark.parametrize("uri", ["data.csv", "data.parquet", "data.xlsx"])
    def test_infer_default_driver_is_pandas(self, uri: str):
        # A DataFrame source without an explicit driver must default to a
        # DataFrame driver ("pandas"), not another source type's driver such
        # as "geodataframe_table", which also claims .csv/.parquet (#1403).
        assert DataFrameSource._infer_default_driver(uri) == "pandas"

    def test_model_validate_without_driver_defaults_to_pandas(self):
        source = DataFrameSource.model_validate(
            {"name": "df", "data_type": "DataFrame", "uri": "data.csv"}
        )
        assert source.driver.name == "pandas"
