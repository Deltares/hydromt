"""Tests the netcdf driver."""

import xarray as xr
from pytest_mock import MockerFixture
from xarray import open_mfdataset

from hydromt.drivers import NetcdfDriver
from hydromt.drivers.preprocessing import round_latlon


class TestNetcdfDriver:
    def test_calls_preprocess(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.drivers.netcdf_driver.xr.open_mfdataset", spec=open_mfdataset
        )
        mock_xr_open.return_value = xr.Dataset()
        uris: str = "dir_with_netcdfs/{variable}.netcdf"
        res: xr.Dataset = NetcdfDriver().read(
            uris, preprocess="round_latlon", variables=["var1", "var2"]
        )
        call_args = mock_xr_open.call_args
        assert call_args[0][0] == [
            "dir_with_netcdfs/var1.netcdf",
            "dir_with_netcdfs/var2.netcdf",
        ]  # first arg
        assert call_args[1].get("preprocess") == round_latlon
        assert res.sizes == {}  # empty dataframe
