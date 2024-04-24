"""Tests the netcdf driver."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import xarray as xr
from pytest_mock import MockerFixture
from xarray import open_mfdataset

from hydromt.drivers import NetcdfDriver
from hydromt.drivers.preprocessing import round_latlon
from hydromt.metadata_resolver.metadata_resolver import MetaDataResolver


class TestNetcdfDriver:
    def test_calls_preprocess(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.drivers.netcdf_driver.xr.open_mfdataset", spec=open_mfdataset
        )
        mock_xr_open.return_value = xr.Dataset()

        class FakeMetadataResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return [uri]

        uri: str = "file.netcdf"
        driver = NetcdfDriver(
            metadata_resolver=FakeMetadataResolver(),
            options={"preprocess": "round_latlon"},
        )
        res: xr.Dataset = driver.read(
            uri,
            variables=["var1", "var2"],
        )
        call_args = mock_xr_open.call_args
        assert call_args[0][0] == [uri]  # first arg
        assert call_args[1].get("preprocess") == round_latlon
        assert res.sizes == {}  # empty dataframe

        assert (
            driver.options.get("preprocess") == "round_latlon"
        )  # test does not consume property

    def test_write(self, raster_ds: xr.Dataset, tmp_path: Path):
        netcdf_path = tmp_path / f"{uuid4().hex}.nc"
        driver = NetcdfDriver()
        driver.write(netcdf_path, raster_ds)
        assert np.all(driver.read(str(netcdf_path)) == raster_ds)
