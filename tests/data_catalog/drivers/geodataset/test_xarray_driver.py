"""Tests the GeoDatasetXarray driver."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import xarray as xr
from pytest_mock import MockerFixture
from xarray import open_mfdataset

from hydromt.data_catalog.drivers.geodataset.xarray_driver import GeoDatasetXarrayDriver
from hydromt.data_catalog.drivers.preprocessing import round_latlon
from hydromt.data_catalog.uri_resolvers.convention_resolver import ConventionResolver
from hydromt.data_catalog.uri_resolvers.uri_resolver import URIResolver


class TestGeoDatasetXarrayDriver:
    def test_calls_preprocess(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.xarray_driver.xr.open_mfdataset",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        class FakeURIResolver(URIResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return [uri]

        uri: str = "file.netcdf"
        driver = GeoDatasetXarrayDriver(
            uri_resolver=FakeURIResolver(),
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

    def test_write(self, geods: xr.Dataset, tmp_path: Path):
        netcdf_path = tmp_path / f"{uuid4().hex}.nc"
        driver = GeoDatasetXarrayDriver()
        driver.write(netcdf_path, geods)
        assert np.all(driver.read(str(netcdf_path)) == geods)

    def test_zarr_read(self, example_zarr_file: Path):
        res: xr.Dataset = GeoDatasetXarrayDriver(
            uri_resolver=ConventionResolver()
        ).read(str(example_zarr_file))
        assert list(res.data_vars.keys()) == ["variable"]
        assert res["variable"].shape == (10, 10)
        assert list(res.coords.keys()) == ["xc", "yc"]
        assert res["variable"].values[0, 0] == 42

    def test_zarr_write(self, geods: xr.Dataset, tmp_dir: Path):
        zarr_path: Path = tmp_dir / "geo.zarr"
        driver = GeoDatasetXarrayDriver()
        driver.write(zarr_path, geods)
        assert np.all(driver.read(str(zarr_path)) == geods)

    def test_calls_zarr_with_zarr_ext(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.xarray_driver.xr.open_zarr",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        class FakeURIResolver(URIResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return [uri]

        uri: str = "file.zarr"
        driver = GeoDatasetXarrayDriver(uri_resolver=FakeURIResolver())
        _ = driver.read(uri)
        assert mock_xr_open.call_count == 1

    def test_calls_nc_func_with_nc_ext(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.raster.raster_xarray_driver.xr.open_mfdataset",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        class FakeURIResolver(URIResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return [uri]

        uri: str = "file.netcdf"
        driver = GeoDatasetXarrayDriver(uri_resolver=FakeURIResolver())
        _ = driver.read(uri)
        assert mock_xr_open.call_count == 1
