import glob
from os.path import join
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import rasterio
import xarray as xr

from hydromt.drivers.rasterio_driver import open_mfraster, open_raster
from hydromt.gis.raster import full_from_transform


class TestOpenMFRaster:
    @pytest.fixture()
    def raster_file(self, tmp_dir: Path, rioda: xr.DataArray) -> str:
        uri_tif = str(tmp_dir / "test.tif")
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

    @pytest.fixture()
    def raster_file_masked_windowed(self, tmp_dir: Path, raster_file: str) -> str:
        uri_tif = str(tmp_dir / "test_masked.tif")
        da_masked = open_raster(raster_file, mask_nodata=True)
        da_masked.raster.to_raster(uri_tif, nodata=-9999, windowed=True)
        return uri_tif

    def test_open_raster_windowed_nodata(self, raster_file_masked_windowed: str):
        # TODO window needs checking & better testing
        da_windowed = open_raster(raster_file_masked_windowed)
        assert not np.any(np.isnan(da_windowed.values))

    @pytest.fixture()
    def raster_file_t_dim(self, tmp_dir: Path, raster_file_masked_windowed: str) -> str:
        uri_tif = str(tmp_dir / "t_dim.tif")
        da_masked = open_raster(raster_file_masked_windowed)
        da_masked.fillna(da_masked.attrs["_FillValue"]).expand_dims("t").round(
            0
        ).astype(np.int32).raster.to_raster(uri_tif, dtype=np.int32)
        return uri_tif

    def test_open_raster_t_dim(self, raster_file_t_dim: str):
        da_windowed = open_raster(raster_file_t_dim)
        assert da_windowed.dtype == np.int32

    @pytest.fixture()
    def raster_mapstack(
        self, tmp_dir: Path, rioda: xr.DataArray
    ) -> Tuple[str, str, xr.Dataset]:
        ds = rioda.to_dataset()
        prefix = "_test_"
        root = str(tmp_dir)
        ds.raster.to_mapstack(root, prefix=prefix, mask=True, driver="GTiff")
        return root, prefix, ds

    def test_open_mfraster_mapstack(self, raster_mapstack: Tuple[str, str, xr.Dataset]):
        root, prefix, ds = raster_mapstack
        ds_in = open_mfraster(join(root, f"{prefix}*.tif"), mask_nodata=True)
        dvars = ds_in.raster.vars
        assert np.all([n in dvars for n in ds.raster.vars])
        assert np.all([np.isnan(ds_in[n].raster.nodata) for n in dvars])

    @pytest.fixture()
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
        ds.rename({"test": "test/test"}).raster.to_mapstack(root, driver="GTiff")
        assert (Path(root) / "test" / "test.tif").is_file()

    def test_open_mfraster_not_found(self, tmp_dir: Path):
        with pytest.raises(OSError, match="no files to open"):
            open_mfraster(str(tmp_dir / "test*.tiffff"))

    def test_open_mfraster_mergeerror(self, tmp_dir: Path):
        da0: xr.DataArray = full_from_transform(
            [0.5, 0.0, 3.0, 0.0, -0.5, -9.0], (4, 6), nodata=-1, name="test"
        )
        da1: xr.DataArray = full_from_transform(
            [0.2, 0.0, 3.0, 0.0, 0.25, -11.0], (8, 15), nodata=-1, name="test"
        )
        da0.raster.to_raster(str(tmp_dir / "test0.tif"))
        da1.raster.to_raster(str(tmp_dir / "test1.tif"))
        with pytest.raises(
            xr.MergeError, match="Geotransform and/or shape do not match"
        ):
            open_mfraster(str(tmp_dir / "test*.tif"))
