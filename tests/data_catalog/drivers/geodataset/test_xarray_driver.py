"""Tests the GeoDatasetXarray driver."""

from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
import xarray as xr
from pytest_mock import MockerFixture
from xarray import open_mfdataset

from hydromt.data_catalog.drivers.geodataset.xarray_driver import GeoDatasetXarrayDriver
from hydromt.data_catalog.drivers.preprocessing import round_latlon


class TestGeoDatasetXarrayDriver:
    def test_calls_preprocess(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.xarray_driver.xr.open_mfdataset",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        uris: List[str] = ["file.netcdf"]
        driver = GeoDatasetXarrayDriver(
            options={"preprocess": "round_latlon"},
        )
        res: xr.Dataset = driver.read(uris)
        call_args = mock_xr_open.call_args
        assert call_args[0][0] == uris  # first arg
        assert call_args[1].get("preprocess") == round_latlon
        assert res.sizes == {}  # empty dataframe

        assert driver.options.preprocess == "round_latlon"

    def test_write(self, geods: xr.Dataset, tmp_path: Path):
        netcdf_path = tmp_path / f"{uuid4().hex}.nc"
        driver = GeoDatasetXarrayDriver()
        driver.write(netcdf_path, geods)
        assert np.all(driver.read([str(netcdf_path)]) == geods)

    def test_zarr_read(self, example_zarr_file: Path):
        res: xr.Dataset = GeoDatasetXarrayDriver().read([str(example_zarr_file)])
        assert list(res.data_vars.keys()) == ["variable"]
        assert res["variable"].shape == (10, 10)
        assert list(res.coords.keys()) == ["xc", "yc"]
        assert res["variable"].values[0, 0] == 42

    def test_zarr_write(self, geods: xr.Dataset, managed_tmp_path: Path):
        zarr_path: Path = managed_tmp_path / "geo.zarr"
        driver = GeoDatasetXarrayDriver()
        driver.write(zarr_path, geods)
        assert np.all(driver.read([str(zarr_path)]) == geods)

    def test_calls_zarr_with_zarr_ext(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.xarray_driver.xr.open_zarr",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        uris: List[str] = ["file.zarr"]
        driver = GeoDatasetXarrayDriver()
        _ = driver.read(uris)
        assert mock_xr_open.call_count == 1

    def test_calls_nc_func_with_nc_ext(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.xarray_driver.xr.open_mfdataset",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        uris: List[str] = ["file.netcdf"]
        driver = GeoDatasetXarrayDriver()
        _ = driver.read(uris)
        assert mock_xr_open.call_count == 1

    def test_calls_nc_func_with_nc_ext_override(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.xarray_driver.xr.open_mfdataset",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        uris: List[str] = ["file.nc", "file.netcdf", "file.something", "file.else"]
        driver = GeoDatasetXarrayDriver()

        # No override should read only .nc
        _ = driver.read(uris)
        assert mock_xr_open.call_count == 1
        assert mock_xr_open.call_args[0][0] == ["file.nc", "file.netcdf"]

        # With override should read all as .nc
        mock_xr_open.reset_mock()
        driver.options.ext_override = ".nc"
        _ = driver.read(uris)
        assert mock_xr_open.call_count == 1
        assert mock_xr_open.call_args[0][0] == uris

    def test_calls_zarr_func_with_zarr_ext_override(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.xarray_driver.xr.open_zarr",
            spec=xr.open_zarr,
        )
        mock_xr_open.return_value = xr.Dataset()

        uris: List[str] = [
            "file.zarr",
            "file.nc",
            "file.netcdf",
            "file.something",
            "file.else",
        ]
        driver = GeoDatasetXarrayDriver()

        # No override should read only .zarr
        _ = driver.read(uris)

        assert mock_xr_open.call_count == 1
        called_uris = [call.args[0] for call in mock_xr_open.call_args_list]
        assert called_uris == ["file.zarr"]

        # With override should read all as .zarr
        mock_xr_open.reset_mock()
        driver.options.ext_override = ".zarr"
        _ = driver.read(uris)

        assert mock_xr_open.call_count == len(uris)
        called_uris = [call.args[0] for call in mock_xr_open.call_args_list]
        assert called_uris == uris
