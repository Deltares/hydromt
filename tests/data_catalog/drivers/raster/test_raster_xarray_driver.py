"""Tests the RasterXarray driver."""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import xarray as xr
from pytest_mock import MockerFixture
from xarray import open_mfdataset

from hydromt.data_catalog.data_catalog import DataCatalog
from hydromt.data_catalog.drivers.preprocessing import round_latlon
from hydromt.data_catalog.drivers.raster.raster_xarray_driver import (
    RasterDatasetXarrayDriver,
    RasterXarrayOptions,
)


def test_driver_options():
    """Test that new driver options are correctly parsed from a dictionary.

    I used the contents of tests/data/test_sources1.yaml here to create the dict.
    """
    options = {
        "chunks": {
            "time": 100,
            "longitude": 120,
            "latitude": 125,
        },
        "concat_dim": "time",
        "decode_times": True,
        "combine": "by_coords",
        "parallel": True,
    }

    data_dict = {
        "era5": {
            "data_type": "RasterDataset",
            "uri": (Path.home() / ".hydromt/artifact_data/latest/era5.nc").as_posix(),
            "driver": {
                "name": "raster_xarray",
                "options": options,
            },
        },
    }

    dc = DataCatalog().from_dict(data_dict=data_dict)
    read_options = dc.get_source("era5").driver.options

    assert isinstance(read_options, RasterXarrayOptions)
    for option in options:
        assert getattr(read_options, option) == options[option]


class TestRasterXarrayDriver:
    def test_calls_preprocess(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.raster.raster_xarray_driver.xr.open_mfdataset",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        uris: List[str] = ["file.netcdf"]
        driver = RasterDatasetXarrayDriver(
            options={"preprocess": "round_latlon"},
        )
        res: xr.Dataset = driver.read(uris)
        call_args = mock_xr_open.call_args
        assert call_args[0][0] == uris  # first arg
        assert call_args[1].get("preprocess") == round_latlon
        assert res.sizes == {}  # empty dataframe

        assert driver.options.preprocess == "round_latlon"
        # test does not consume property

    def test_write(self, raster_ds: xr.Dataset, tmp_path: Path):
        netcdf_path = tmp_path / f"{uuid4().hex}.nc"
        driver = RasterDatasetXarrayDriver()
        driver.write(netcdf_path, raster_ds)
        assert np.all(driver.read([str(netcdf_path)]) == raster_ds)

    def test_unknown_ext(self):
        driver = RasterDatasetXarrayDriver()
        mock_ds = MagicMock()
        gpkg_path: Path = Path("path") / "to" / "file.gpkg"
        driver.write(gpkg_path, mock_ds)
        mock_ds.to_zarr.assert_called_once()

    def test_zarr_read(self, example_zarr_file: Path):
        res: xr.Dataset = RasterDatasetXarrayDriver().read([str(example_zarr_file)])
        assert list(res.data_vars.keys()) == ["variable"]
        assert res["variable"].shape == (10, 10)
        assert list(res.coords.keys()) == ["xc", "yc"]
        assert res["variable"].values[0, 0] == 42

    def test_zarr_write(self, raster_ds: xr.Dataset, managed_tmp_path: Path):
        zarr_path = managed_tmp_path / "raster.zarr"
        driver = RasterDatasetXarrayDriver()
        driver.write(zarr_path, raster_ds)
        assert np.all(driver.read([str(zarr_path)]) == raster_ds)

    def test_calls_zarr_with_zarr_ext(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.raster.raster_xarray_driver.xr.open_zarr",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        uris: List[str] = ["file.zarr"]
        driver = RasterDatasetXarrayDriver()
        _ = driver.read(uris)
        assert mock_xr_open.call_count == 1

    def test_calls_nc_func_with_nc_ext(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.raster.raster_xarray_driver.xr.open_mfdataset",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        uris: List[str] = ["file.netcdf"]
        driver = RasterDatasetXarrayDriver()
        _ = driver.read(uris)
        assert mock_xr_open.call_count == 1
