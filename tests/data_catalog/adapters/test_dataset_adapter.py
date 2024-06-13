import numpy as np
import pytest
import xarray as xr

from hydromt._typing import NoDataException, NoDataStrategy, SourceMetadata
from hydromt.data_catalog.adapters.dataset import DatasetAdapter


class TestRasterDatasetAdapter:
    def test_transform_nodata(self, timeseries_ds: xr.Dataset):
        adapter = DatasetAdapter()
        time_range = ("01-01-1970", "01-01-1971")
        with pytest.raises(NoDataException):
            adapter.transform(
                timeseries_ds, metadata=SourceMetadata(), time_range=time_range
            )
        ds = adapter.transform(
            timeseries_ds,
            metadata=SourceMetadata(),
            time_range=time_range,
            handle_nodata=NoDataStrategy.IGNORE,
        )
        assert ds is None

    def test_set_nodata(self, timeseries_ds: xr.Dataset):
        nodata = -999
        dataset_adapter = DatasetAdapter()
        ds = dataset_adapter.transform(
            timeseries_ds, metadata=SourceMetadata(nodata=nodata)
        )
        for k in ds.data_vars:
            assert ds[k].attrs["_FillValue"] == nodata

        nodata = {"col1": -999, "col2": np.nan}
        ds = dataset_adapter.transform(
            timeseries_ds, metadata=SourceMetadata(nodata=nodata)
        )
        assert np.isnan(ds["col2"].attrs["_FillValue"])
        assert ds["col1"].attrs["_FillValue"] == nodata["col1"]

    def test_apply_unit_conversion(self, timeseries_ds: xr.Dataset):
        dataset_adapter = DatasetAdapter(
            unit_mult=dict(col1=1000),
        )
        ds1 = dataset_adapter.transform(timeseries_ds, SourceMetadata())
        assert ds1["col1"].equals(timeseries_ds["col1"] * 1000)

        dataset_adapter = DatasetAdapter(unit_add={"time": 10})
        ds2 = dataset_adapter.transform(timeseries_ds, SourceMetadata())
        assert ds2["time"][-1].values == np.datetime64("2020-12-31T00:00:10")

    def test_dataset_set_metadata(self, timeseries_ds: xr.Dataset):
        meta_data = {"col1": {"long_name": "column1"}, "col2": {"long_name": "column2"}}
        dataset_adapter = DatasetAdapter()
        ds = dataset_adapter.transform(timeseries_ds, SourceMetadata(attrs=meta_data))
        assert ds["col1"].attrs["long_name"] == "column1"
        assert ds["col2"].attrs["long_name"] == "column2"
