from copy import copy

import numpy as np
import pandas as pd

from hydromt.data_adapter.dataframe import DataFrameAdapter
from hydromt.data_source import SourceMetadata


class TestDataFrameAdapter:
    def test_transform_no_filters_no_meta(self, df: pd.DataFrame):
        adapter = DataFrameAdapter()
        metadata = SourceMetadata()
        res = adapter.transform(df, metadata)
        assert np.all(res == df)

    def test_transform_variables(self, df: pd.DataFrame):
        adapter = DataFrameAdapter(unit_add={"latitude": 1, "longitude": -1})
        metadata = SourceMetadata()
        df_copy = copy(df)
        res = adapter.transform(df, metadata)
        assert np.all(res["longitude"] == df_copy["longitude"] - 1)
        assert np.all(res["latitude"] == df_copy["latitude"] + 1)

    def test_transform_meta(self, df: pd.DataFrame):
        adapter = DataFrameAdapter()
        metadata = SourceMetadata(
            attrs={"longitude": {"attr1": 1}}, url="www.example.com"
        )
        res = adapter.transform(df, metadata)
        assert res["longitude"].attrs["attr1"] == 1
        assert res.attrs["url"] == "www.example.com"
