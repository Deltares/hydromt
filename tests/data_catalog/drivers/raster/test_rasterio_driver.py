import glob
import shutil
from os.path import join
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
import xarray as xr

from hydromt._compat import HAS_GDAL
from hydromt.data_catalog.drivers.raster.rasterio_driver import (
    RasterioDriver,
    RasterioOptions,
)
from hydromt.gis.raster_utils import full_from_transform
from hydromt.readers import open_mfraster, open_raster
from hydromt.typing import SourceMetadata
from tests.conftest import TEST_DATA_DIR


class TestRasterioDriver:
    @pytest.fixture
    def vrt_tiled_raster_ds(self, tmp_path: Path) -> str:
        # copy vrt data to test folder
        name = "test_vrt_tiled_raster_ds"
        root = tmp_path / name
        shutil.copytree(join(TEST_DATA_DIR, "rioda_tiled"), root)
        return str(root)

    @pytest.mark.skipif(not HAS_GDAL, reason="GDAL not installed.")
    @pytest.mark.usefixtures("test_settings")
    def test_caches_tifs_from_vrt(self, vrt_tiled_raster_ds: str):
        cache_dir: str = "tests_caches_tifs_from_vrt"
        options = RasterioOptions(cache=True, cache_dir=cache_dir)
        driver = RasterioDriver(options=options)
        uris = [join(vrt_tiled_raster_ds, "tiled_zl0.vrt")]
        driver.read(uris=uris)
        cache_path = options.get_cache_path(uris)
        assert len(list(cache_path.glob("**/*.tif"))) == 16

    @pytest.fixture
    def small_tif(self, tmp_path: Path, rioda: xr.DataArray) -> str:
        path = tmp_path / "small_tif.tif"
        rioda.raster.to_raster(str(path))
        return str(path)

    def test_reads(self, small_tif: str, rioda: xr.DataArray):
        ds: xr.Dataset = RasterioDriver().read(small_tif)
        xr.testing.assert_equal(rioda, ds["small_tif"])

    def test_renames_single_var(self, small_tif: str):
        ds: xr.Dataset = RasterioDriver().read([small_tif], variables=["test"])
        assert list(ds.data_vars) == ["test"]

    def test_sets_nodata(self, rioda: xr.DataArray, tmp_path: Path):
        # hard-reset nodata value of rioda (cannot use set_nodata to set None)
        rioda.rio.set_nodata(None, inplace=True)
        rioda.rio.write_nodata(None, inplace=True)
        rioda.raster.set_nodata(None)
        uri: str = str(tmp_path / "test_sets_nodata.tif")
        rioda.raster.to_raster(uri)
        ds: xr.Dataset = RasterioDriver().read(
            uri, metadata=SourceMetadata(nodata=np.float64(42))
        )
        assert ds["test_sets_nodata"].raster.nodata == 42

    @patch("hydromt.data_catalog.drivers.raster.rasterio_driver.open_mfraster")
    def test_sets_mosaic_kwargs(self, fake_open_mfraster: MagicMock):
        uris = ["test", "test2"]
        mosaic_kwargs = {"mykwarg": 0}
        options_dict = {"mosaic_kwargs": mosaic_kwargs}
        options = RasterioOptions(**options_dict)
        RasterioDriver(options=options).read(uris=uris)
        fake_open_mfraster.assert_called_once_with(
            uris, mosaic=False, mosaic_kwargs=mosaic_kwargs
        )

    def test_write(self, rioda: xr.DataArray, tmp_path: Path):
        uri: str = tmp_path / "test_write.tif"
        p = RasterioDriver().write(uri, rioda)
        assert isinstance(p, Path)
        assert uri.is_file()
        uri_unknown_extension = tmp_path / "test_write.raster"
        with pytest.raises(
            ValueError, match="Unknown extension for RasterioDriver: .raster"
        ):
            RasterioDriver().write(uri_unknown_extension, rioda)

    def test_write_wildcard(self, rioda: xr.DataArray, tmp_path: Path):
        uri: str = tmp_path / "test_write_*.tif"
        with pytest.raises(
            ValueError,
            match="Writing multiple files with wildcard requires at least 3 dimensions in data array",
        ):
            RasterioDriver().write(uri, rioda)

        # expand dims to have at least 3 dimensions
        rioda_expanded = rioda.expand_dims("time").assign_coords(time=[0])
        p = RasterioDriver().write(uri, rioda_expanded)
        written = list(uri.parent.glob(uri.name))
        assert len(written) == 1
        assert p == uri

    def test_write_dataset(self, rioda: xr.DataArray, tmp_path: Path):
        ds = rioda.to_dataset(name="var1")
        uri: str = tmp_path / "test_write_dataset.tif"
        p = RasterioDriver().write(uri, ds)
        assert p == tmp_path / "var1.tif"

        # Test with two variables
        ds["var2"] = rioda + 10
        uri: str = tmp_path / "test_write_dataset2.tif"
        p = RasterioDriver().write(uri, ds)
        assert p == tmp_path / "*.tif"
        tif_files = list((tmp_path).glob("*.tif"))
        assert len(tif_files) == 2


class TestOpenMFRaster:
    @pytest.fixture
    def raster_file(self, tmp_path: Path, rioda: xr.DataArray) -> str:
        uri_tif = str(tmp_path / "test_open_mfraster.tif")
        rioda.raster.to_raster(uri_tif, crs=3857, tags={"name": "test"})
        return uri_tif

    def test_open_raster(self, raster_file: str, rioda: xr.DataArray):
        assert np.all(open_raster(raster_file).values == rioda.values)
        with rasterio.open(raster_file, "r") as src:
            assert src.tags()["name"] == "test"
            assert src.crs.to_epsg() == 3857

    def test_open_raster_mask_nodata(self, raster_file: str):
        da_masked = open_raster(raster_file, mask_nodata=True)
        assert np.any(np.isnan(da_masked.values))

    @pytest.fixture
    def raster_file_masked_windowed(self, tmp_path: Path, raster_file: str) -> str:
        uri_tif = str(tmp_path / "test_masked.tif")
        da_masked = open_raster(raster_file, mask_nodata=True)
        da_masked.raster.to_raster(uri_tif, nodata=-9999, windowed=True)
        return uri_tif

    def test_open_raster_windowed_nodata(self, raster_file_masked_windowed: str):
        da_windowed = open_raster(raster_file_masked_windowed)
        assert not np.any(np.isnan(da_windowed.values))

    @pytest.fixture
    def raster_file_t_dim(
        self, tmp_path: Path, raster_file_masked_windowed: str
    ) -> str:
        uri_tif = str(tmp_path / "t_dim.tif")
        da_masked = open_raster(raster_file_masked_windowed)
        da_masked.fillna(da_masked.attrs["_FillValue"]).expand_dims("t").round(
            0
        ).astype(np.int32).raster.to_raster(uri_tif, dtype=np.int32)
        return uri_tif

    def test_open_raster_t_dim(self, raster_file_t_dim: str):
        da_windowed = open_raster(raster_file_t_dim)
        assert da_windowed.dtype == np.int32

    @pytest.fixture
    def raster_mapstack(
        self, tmp_path: Path, rioda: xr.DataArray
    ) -> Tuple[str, str, xr.Dataset]:
        ds = rioda.to_dataset()
        prefix = "_test_"
        root = tmp_path
        ds.raster.to_mapstack(root, prefix=prefix, mask=True, driver="GTiff")
        return root, prefix, ds

    def test_open_mfraster_mapstack(self, raster_mapstack: Tuple[str, str, xr.Dataset]):
        root, prefix, ds = raster_mapstack
        ds_in = open_mfraster(join(root, f"{prefix}*.tif"), mask_nodata=True)
        dvars = ds_in.raster.vars
        assert np.all([n in dvars for n in ds.raster.vars])
        assert np.all([np.isnan(ds_in[n].raster.nodata) for n in dvars])

    @pytest.fixture
    def raster_mapstack_plus_one(
        self, raster_mapstack: Tuple[str, str, xr.Dataset], rioda: xr.DataArray
    ) -> Tuple[str, str, xr.Dataset]:
        root, prefix, ds = raster_mapstack
        # concat new tif
        uri_tif = Path(root) / "test_3.tif"
        rioda.raster.to_raster(uri_tif, crs=3857)
        return root, prefix, ds

    def test_open_mfraster(self, raster_mapstack_plus_one: Tuple[str, str, xr.Dataset]):
        root, _, _ = raster_mapstack_plus_one
        ds_in = open_mfraster(str(Path(root) / "test_*.tif"), concat=True)
        assert ds_in[ds_in.raster.vars[0]].ndim == 3

    def test_open_mfraster_paths(
        self, raster_mapstack_plus_one: Tuple[str, str, xr.Dataset]
    ):
        root, prefix, ds = raster_mapstack_plus_one
        # with reading with pathlib
        paths = [Path(p) for p in glob.glob(join(root, f"{prefix}*.tif"))]
        dvars2 = open_mfraster(paths, mask_nodata=True).raster.vars
        assert np.all([f"{prefix}{n}" in dvars2 for n in ds.raster.vars])
        # test writing to subdir
        new_name: str = "test_open_mfraster_paths"
        ds.rename({"test": f"test/{new_name}"}).raster.to_mapstack(root, driver="GTiff")
        assert (Path(root) / "test" / f"{new_name}.tif").is_file()

    def test_open_mfraster_not_found(self, tmp_path: Path):
        with pytest.raises(OSError, match="no files to open"):
            open_mfraster(str(tmp_path / "test*.tiffff"))

    def test_open_mfraster_mergeerror(self, tmp_path: Path):
        da0: xr.DataArray = full_from_transform(
            [0.5, 0.0, 3.0, 0.0, -0.5, -9.0], (4, 6), nodata=-1, name="test"
        )
        da1: xr.DataArray = full_from_transform(
            [0.2, 0.0, 3.0, 0.0, 0.25, -11.0], (8, 15), nodata=-1, name="test"
        )
        da0.raster.to_raster(str(tmp_path / "test0.tif"))
        da1.raster.to_raster(str(tmp_path / "test1.tif"))
        with pytest.raises(
            xr.MergeError, match="Geotransform and/or shape do not match"
        ):
            open_mfraster(str(tmp_path / "test*.tif"))
