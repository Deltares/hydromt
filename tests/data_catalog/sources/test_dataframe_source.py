from pathlib import Path
from typing import Type

import pandas as pd

from hydromt.data_catalog.adapters import DataFrameAdapter
from hydromt.data_catalog.drivers import DataFrameDriver
from hydromt.data_catalog.sources import DataFrameSource


class TestDataFrameSource:
    def test_read_data(
        self,
        MockDataFrameDriver: Type[DataFrameDriver],
        mock_df_adapter: DataFrameAdapter,
        df: pd.DataFrame,
        tmp_dir: Path,
    ):
        tmp_dir.touch("test.xls")
        source = DataFrameSource(
            root=".",
            name="example_source",
            driver=MockDataFrameDriver(),
            data_adapter=mock_df_adapter,
            uri=str(tmp_dir / "test.xls"),
        )
        pd.testing.assert_frame_equal(df, source.read_data())