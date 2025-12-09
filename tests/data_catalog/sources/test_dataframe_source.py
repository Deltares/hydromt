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
