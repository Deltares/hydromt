"""Tests the netcdf driver."""
from typing import List

import xarray as xr
from pytest_mock import MockerFixture
from xarray import open_mfdataset

from hydromt.driver import NetcdfDriver
from hydromt.driver.preprocessing import round_latlon


class TestNetcdfDriver:
    def test_calls_preprocess(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.driver.netcdf_driver.xr.open_mfdataset", spec=open_mfdataset
        )
        mock_xr_open.return_value = xr.Dataset()
        uris: List[str] = ["some", "uris"]
        res: xr.Dataset = NetcdfDriver().read(uris, preprocess="round_latlon")
        call_args = mock_xr_open.call_args
        assert call_args[0][0] == uris  # first arg
        assert call_args[1].get("preprocess") == round_latlon
        assert res.sizes == {}  # empty dataframe
