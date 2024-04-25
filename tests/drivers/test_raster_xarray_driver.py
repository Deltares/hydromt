"""Tests the netcdf driver."""

from pathlib import Path
from typing import Tuple
from uuid import uuid4

import numpy as np
import pytest
import xarray as xr
import zarr
from pytest_mock import MockerFixture
from xarray import open_mfdataset

from hydromt.drivers.preprocessing import round_latlon
from hydromt.drivers.raster_xarray_driver import RasterDatasetXarrayDriver
from hydromt.metadata_resolver.convention_resolver import ConventionResolver
from hydromt.metadata_resolver.metadata_resolver import MetaDataResolver


class TestRasterXarrayDriver:
    def test_calls_preprocess(self, mocker: MockerFixture):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.drivers.raster_xarray_driver.xr.open_mfdataset",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        class FakeMetadataResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return [uri]

        uri: str = "file.netcdf"
        driver = RasterDatasetXarrayDriver(
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

    def test_write(self, rasterds: xr.Dataset, tmp_path: Path):
        netcdf_path = tmp_path / f"{uuid4().hex}.nc"
        driver = RasterDatasetXarrayDriver()
        driver.write(netcdf_path, rasterds)
        assert np.all(driver.read(str(netcdf_path)) == rasterds)

    @pytest.fixture()
    def example_zarr_file(self, tmp_dir: Path) -> Tuple[zarr.Group, Path]:
        tmp_path: Path = tmp_dir / "0s.zarr"
        store = zarr.DirectoryStore(tmp_path)
        root: zarr.Group = zarr.group(store=store)
        zarray_var: zarr.Array = root.zeros("variable", shape=(10, 10), chunks=(5, 5))
        zarray_var.attrs.update(
            {
                "_ARRAY_DIMENSIONS": ["x", "y"],
                "coordinates": "xc yc",
                "long_name": "Test Array",
                "type_preferred": "int8",
            }
        )
        zarray_x: zarr.Array = root.array(
            "xc",
            np.arange(0, 10, dtype=np.dtypes.Int8DType),
            chunks=(5,),
            dtype="int8",
        )
        zarray_x.attrs["_ARRAY_DIMENSIONS"] = ["x"]
        zarray_y: zarr.Array = root.array(
            "yc", np.arange(0, 10, dtype=np.dtypes.Int8DType), chunks=(5,), dtype="int8"
        )
        zarray_y.attrs["_ARRAY_DIMENSIONS"] = ["y"]
        zarr.consolidate_metadata(store)
        return (root, tmp_path)

    def test_zarr_read(self, example_zarr_file: Tuple[zarr.Group, Path]):
        assert (
            RasterDatasetXarrayDriver(metadata_resolver=ConventionResolver()).read(
                str(example_zarr_file[1])
            )
            == example_zarr_file[0]
        )

    def test_zarr_write(self, rasterds: xr.Dataset, tmp_dir: Path):
        zarr_path: Path = tmp_dir / "raster.zarr"
        driver = RasterDatasetXarrayDriver()
        driver.write(zarr_path, rasterds)
        assert np.all(driver.read(str(zarr_path)) == rasterds)
