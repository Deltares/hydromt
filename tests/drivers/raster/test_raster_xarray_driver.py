"""Tests the RasterXarray driver."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import xarray as xr
import zarr
from pytest_mock import MockerFixture
from xarray import open_mfdataset

from hydromt.data_source import SourceMetadata
from hydromt.drivers.preprocessing import round_latlon
from hydromt.drivers.raster.raster_xarray_driver import RasterDatasetXarrayDriver
from hydromt.metadata_resolver.convention_resolver import ConventionResolver
from hydromt.metadata_resolver.metadata_resolver import MetaDataResolver


class TestRasterXarrayDriver:
    @pytest.fixture()
    def metadata(self):
        return SourceMetadata()

    def test_calls_preprocess(self, mocker: MockerFixture, metadata: SourceMetadata):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.drivers.raster.raster_xarray_driver.xr.open_mfdataset",
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
            metadata,
            variables=["var1", "var2"],
        )
        call_args = mock_xr_open.call_args
        assert call_args[0][0] == [uri]  # first arg
        assert call_args[1].get("preprocess") == round_latlon
        assert res.sizes == {}  # empty dataframe

        assert (
            driver.options.get("preprocess") == "round_latlon"
        )  # test does not consume property

    def test_write(
        self, raster_ds: xr.Dataset, tmp_path: Path, metadata: SourceMetadata
    ):
        netcdf_path = tmp_path / f"{uuid4().hex}.nc"
        driver = RasterDatasetXarrayDriver()
        driver.write(netcdf_path, raster_ds)
        assert np.all(driver.read(str(netcdf_path), metadata) == raster_ds)

    @pytest.fixture()
    def example_zarr_file(self, tmp_dir: Path) -> Path:
        tmp_path: Path = tmp_dir / "0s.zarr"
        store = zarr.DirectoryStore(tmp_path)
        root: zarr.Group = zarr.group(store=store, overwrite=True)
        zarray_var: zarr.Array = root.zeros(
            "variable", shape=(10, 10), chunks=(5, 5), dtype="int8"
        )
        zarray_var[0, 0] = 42  # trigger write
        zarray_var.attrs.update(
            {
                "_ARRAY_DIMENSIONS": ["x", "y"],
                "coordinates": "xc yc",
                "long_name": "Test Array",
                "type_preferred": "int8",
            }
        )
        # create symmetrical coords
        xy = np.linspace(0, 9, 10, dtype=np.dtypes.Int8DType)
        xcoords, ycoords = np.meshgrid(xy, xy)

        zarray_x: zarr.Array = root.array("xc", xcoords, chunks=(5, 5), dtype="int8")
        zarray_x.attrs["_ARRAY_DIMENSIONS"] = ["x", "y"]
        zarray_y: zarr.Array = root.array("yc", ycoords, chunks=(5, 5), dtype="int8")
        zarray_y.attrs["_ARRAY_DIMENSIONS"] = ["x", "y"]
        zarr.consolidate_metadata(store)
        store.close()
        return tmp_path

    def test_zarr_read(self, example_zarr_file: Path, metadata: SourceMetadata):
        res: xr.Dataset = RasterDatasetXarrayDriver(
            metadata_resolver=ConventionResolver()
        ).read(str(example_zarr_file), metadata)
        assert list(res.data_vars.keys()) == ["variable"]
        assert res["variable"].shape == (10, 10)
        assert list(res.coords.keys()) == ["xc", "yc"]
        assert res["variable"].values[0, 0] == 42

    def test_zarr_write(
        self, raster_ds: xr.Dataset, tmp_dir: Path, metadata: SourceMetadata
    ):
        zarr_path: Path = tmp_dir / "raster.zarr"
        driver = RasterDatasetXarrayDriver()
        driver.write(zarr_path, raster_ds)
        assert np.all(driver.read(str(zarr_path), metadata) == raster_ds)

    def test_calls_zarr_with_zarr_ext(
        self, mocker: MockerFixture, metadata: SourceMetadata
    ):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.drivers.raster.raster_xarray_driver.xr.open_zarr",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        class FakeMetadataResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return [uri]

        uri: str = "file.zarr"
        driver = RasterDatasetXarrayDriver(metadata_resolver=FakeMetadataResolver())
        _ = driver.read(
            uri,
            metadata,
        )
        assert mock_xr_open.call_count == 1

    def test_calls_nc_func_with_nc_ext(
        self, mocker: MockerFixture, metadata: SourceMetadata
    ):
        mock_xr_open: mocker.MagicMock = mocker.patch(
            "hydromt.drivers.raster.raster_xarray_driver.xr.open_mfdataset",
            spec=open_mfdataset,
        )
        mock_xr_open.return_value = xr.Dataset()

        class FakeMetadataResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return [uri]

        uri: str = "file.netcdf"
        driver = RasterDatasetXarrayDriver(metadata_resolver=FakeMetadataResolver())
        _ = driver.read(
            uri,
            metadata,
        )
        assert mock_xr_open.call_count == 1
