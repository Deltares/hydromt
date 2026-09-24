# -*- coding: utf-8 -*-
"""Tests for reading through an fsspec filesystem.

An fsspec ``memory://`` filesystem is a complete stand-in for a remote store,
so these tests cover the remote code paths of the readers without any cloud.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fsspec.implementations.memory import MemoryFileSystem

from hydromt._compat import HAS_H5NETCDF, HAS_H5PY
from hydromt._fsio import is_local
from hydromt.data_catalog.drivers import PandasDriver
from hydromt.data_catalog.drivers.base_driver import resolve_filesystem
from hydromt.readers import (
    open_geodataset,
    open_mfcsv,
    open_mfdataset,
    open_mfraster,
    open_nc,
    open_raster,
    open_raster_from_tindex,
    open_timeseries_from_table,
    open_vector,
    open_vector_from_table,
    open_zarrs,
)


@pytest.fixture
def memory_fs() -> MemoryFileSystem:
    fs = MemoryFileSystem()
    fs.store.clear()
    fs.pseudo_dirs.clear()
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()


def _upload(fs: MemoryFileSystem, local_path: Path, uri: str) -> str:
    """Copy a local file into the memory filesystem and return its uri."""
    fs.mkdirs(uri.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(uri, "wb") as f:
        f.write(Path(local_path).read_bytes())
    return uri


class _CloseCounter:
    """Track how often a handle's ``close`` was called.

    ``MemoryFile.close`` is a no-op, so ``closed`` cannot be inspected directly,
    and wrapping the handle in a proxy breaks xarray's backend detection.
    """

    def __init__(self, handle):
        self.calls = 0
        original_close = handle.close

        def _close():
            self.calls += 1
            return original_close()

        handle.close = _close


def _spy_on_open(
    fs: MemoryFileSystem, monkeypatch: pytest.MonkeyPatch
) -> list[_CloseCounter]:
    """Record every handle opened on ``fs``."""
    counters: list[_CloseCounter] = []
    original_open = fs.open

    def _open(*args, **kwargs):
        handle = original_open(*args, **kwargs)
        counters.append(_CloseCounter(handle))
        return handle

    monkeypatch.setattr(fs, "open", _open)
    return counters


class TestIsLocal:
    def test_none_is_local(self):
        assert is_local(None)

    def test_memory_is_not_local(self, memory_fs: MemoryFileSystem):
        assert not is_local(memory_fs)


class TestOpenRaster:
    def test_reads_through_filesystem(
        self, rioda: xr.DataArray, tmp_path: Path, memory_fs: MemoryFileSystem
    ):
        local_path = tmp_path / "test.tif"
        rioda.raster.to_raster(local_path)
        uri = _upload(memory_fs, local_path, "/raster/test.tif")

        da = open_raster(uri, filesystem=memory_fs)

        assert np.all(da.values == rioda.values)

    def test_stays_lazy_and_readable(
        self, rioda: xr.DataArray, tmp_path: Path, memory_fs: MemoryFileSystem
    ):
        local_path = tmp_path / "test.tif"
        rioda.raster.to_raster(local_path)
        uri = _upload(memory_fs, local_path, "/raster/test.tif")

        da = open_raster(uri, filesystem=memory_fs)

        # not loaded yet, but the backing handle is still open
        assert not da._in_memory
        assert np.all(da.load().values == rioda.values)

    def test_local_filesystem_matches_no_filesystem(
        self, rioda: xr.DataArray, tmp_path: Path
    ):
        local_path = tmp_path / "test.tif"
        rioda.raster.to_raster(local_path)

        da = open_raster(str(local_path))

        assert np.all(da.values == rioda.values)


class TestOpenMfRaster:
    def test_reads_and_globs_through_filesystem(
        self, rioda: xr.DataArray, tmp_path: Path, memory_fs: MemoryFileSystem
    ):
        for name in ("a", "b"):
            local_path = tmp_path / f"{name}.tif"
            rioda.raster.to_raster(local_path)
            _upload(memory_fs, local_path, f"/rasters/{name}.tif")

        ds = open_mfraster("/rasters/*.tif", filesystem=memory_fs)

        assert sorted(ds.data_vars) == ["a", "b"]
        assert np.all(ds["a"].values == rioda.values)

    def test_closes_handles_of_already_opened_files_on_failure(
        self,
        rioda: xr.DataArray,
        tmp_path: Path,
        memory_fs: MemoryFileSystem,
        monkeypatch: pytest.MonkeyPatch,
    ):
        local_path = tmp_path / "a.tif"
        rioda.raster.to_raster(local_path)
        _upload(memory_fs, local_path, "/rasters/a.tif")
        counters = _spy_on_open(memory_fs, monkeypatch)

        with pytest.raises(FileNotFoundError):
            open_mfraster(
                ["/rasters/a.tif", "/rasters/missing.tif"], filesystem=memory_fs
            )

        assert counters
        assert all(counter.calls for counter in counters)

    def test_closes_handles_when_grids_do_not_match(
        self,
        rioda: xr.DataArray,
        tmp_path: Path,
        memory_fs: MemoryFileSystem,
        monkeypatch: pytest.MonkeyPatch,
    ):
        rioda.raster.to_raster(tmp_path / "a.tif")
        _upload(memory_fs, tmp_path / "a.tif", "/rasters/a.tif")
        rioda.isel(x=slice(0, 2)).raster.to_raster(tmp_path / "b.tif")
        _upload(memory_fs, tmp_path / "b.tif", "/rasters/b.tif")
        counters = _spy_on_open(memory_fs, monkeypatch)

        with pytest.raises(xr.MergeError):
            open_mfraster(["/rasters/a.tif", "/rasters/b.tif"], filesystem=memory_fs)

        # the second file's handle is opened before the grid check fails
        assert len(counters) == 2
        assert all(counter.calls for counter in counters)

    def test_closes_handles_with_dataset(
        self,
        rioda: xr.DataArray,
        tmp_path: Path,
        memory_fs: MemoryFileSystem,
        monkeypatch: pytest.MonkeyPatch,
    ):
        local_path = tmp_path / "a.tif"
        rioda.raster.to_raster(local_path)
        _upload(memory_fs, local_path, "/rasters/a.tif")
        counters = _spy_on_open(memory_fs, monkeypatch)

        ds = open_mfraster(["/rasters/a.tif"], filesystem=memory_fs)

        assert counters
        assert not any(counter.calls for counter in counters)
        ds.close()
        assert all(counter.calls for counter in counters)


class TestOpenRasterFromTindex:
    def test_reads_tindex_and_tiles_through_filesystem(
        self, rioda: xr.DataArray, tmp_path: Path, memory_fs: MemoryFileSystem
    ):
        tile_path = tmp_path / "tile.tif"
        rioda.raster.to_raster(tile_path)
        _upload(memory_fs, tile_path, "/tindex/tile.tif")

        tindex = gpd.GeoDataFrame(
            {"location": ["tile.tif"]},
            geometry=[rioda.raster.box.geometry.iloc[0]],
            crs=rioda.raster.crs,
        )
        tindex_path = tmp_path / "tindex.gpkg"
        tindex.to_file(tindex_path, driver="GPKG")
        uri = _upload(memory_fs, tindex_path, "/tindex/tindex.gpkg")

        ds = open_raster_from_tindex(
            uri, geom=rioda.raster.box, filesystem=memory_fs, mask_nodata=True
        )

        assert "tindex" in ds.data_vars
        # clip_geom/rename rebuild the Dataset; the handles must stay closeable
        assert ds._close is not None
        ds.close()


class TestOpenVector:
    def test_reads_ogr_file_through_filesystem(
        self, geodf: gpd.GeoDataFrame, tmp_path: Path, memory_fs: MemoryFileSystem
    ):
        local_path = tmp_path / "test.geojson"
        geodf.to_file(local_path, driver="GeoJSON")
        uri = _upload(memory_fs, local_path, "/vector/test.geojson")

        gdf = open_vector(uri, filesystem=memory_fs)

        assert np.all(gdf == geodf)

    def test_applies_mask_through_filesystem(
        self, geodf: gpd.GeoDataFrame, tmp_path: Path, memory_fs: MemoryFileSystem
    ):
        local_path = tmp_path / "test.geojson"
        geodf.to_file(local_path, driver="GeoJSON")
        uri = _upload(memory_fs, local_path, "/vector/test.geojson")
        mask = gpd.GeoDataFrame(geometry=[geodf.geometry.iloc[0]], crs=geodf.crs)

        gdf = open_vector(uri, geom=mask, filesystem=memory_fs)

        assert gdf.index.size == 1

    def test_reads_table_through_filesystem(
        self, df: pd.DataFrame, geodf: gpd.GeoDataFrame, memory_fs: MemoryFileSystem
    ):
        with memory_fs.open("/vector/test.csv", "wb") as f:
            f.write(df.to_csv().encode())

        gdf = open_vector("/vector/test.csv", crs=4326, filesystem=memory_fs)

        assert np.all(gdf == geodf)

    def test_rejects_remote_multi_file_formats(
        self, geodf: gpd.GeoDataFrame, tmp_path: Path, memory_fs: MemoryFileSystem
    ):
        local_path = tmp_path / "test.shp"
        geodf.to_file(local_path)
        uri = _upload(memory_fs, local_path, "/vector/test.shp")

        with pytest.raises(ValueError, match="needs sidecar files"):
            open_vector(uri, filesystem=memory_fs)


class TestOpenTables:
    def test_open_vector_from_table(
        self, df: pd.DataFrame, geodf: gpd.GeoDataFrame, memory_fs: MemoryFileSystem
    ):
        with memory_fs.open("/tables/test.csv", "wb") as f:
            f.write(df.to_csv().encode())

        gdf = open_vector_from_table("/tables/test.csv", crs=4326, filesystem=memory_fs)

        assert np.all(gdf == geodf)

    def test_open_timeseries_from_table(
        self, ts: pd.DataFrame, memory_fs: MemoryFileSystem
    ):
        with memory_fs.open("/tables/ts.csv", "wb") as f:
            f.write(ts.to_csv().encode())

        da = open_timeseries_from_table("/tables/ts.csv", filesystem=memory_fs)

        # the reader puts time on the first axis
        assert da.shape == (ts.columns.size, ts.index.size)

    def test_open_mfcsv(self, dfs_segmented_by_points, memory_fs: MemoryFileSystem):
        uris = {}
        for i, frame in dfs_segmented_by_points.items():
            uri = f"/tables/{i}.csv"
            with memory_fs.open(uri, "wb") as f:
                f.write(frame.to_csv().encode())
            uris[i] = uri

        ds = open_mfcsv(uris, "id", filesystem=memory_fs)

        assert sorted(ds.id.values) == sorted(dfs_segmented_by_points.keys())


requires_h5_netcdf_and_h5py = pytest.mark.skipif(
    not (HAS_H5NETCDF and HAS_H5PY),
    reason="h5netcdf and h5py are required for this test. Install the `io` extra.",
)


class TestOpenXarray:
    @requires_h5_netcdf_and_h5py
    def test_open_mfdataset_through_filesystem(
        self, obsda: xr.DataArray, tmp_path: Path, memory_fs: MemoryFileSystem
    ):
        local_path = tmp_path / "test.nc"
        obsda.to_dataset(name="test").to_netcdf(local_path)
        uri = _upload(memory_fs, local_path, "/nc/test.nc")

        ds = open_mfdataset([uri], filesystem=memory_fs)

        # lazily backed by the open handle, which is closed with the dataset
        assert np.allclose(ds["test"].values, obsda.values)
        ds.close()

    def test_open_mfdataset_closes_handles_with_dataset(
        self,
        obsda: xr.DataArray,
        tmp_path: Path,
        memory_fs: MemoryFileSystem,
        monkeypatch: pytest.MonkeyPatch,
    ):
        local_path = tmp_path / "test.nc"
        obsda.to_dataset(name="test").to_netcdf(local_path)
        uri = _upload(memory_fs, local_path, "/nc/test.nc")
        counters = _spy_on_open(memory_fs, monkeypatch)

        ds = open_mfdataset([uri], filesystem=memory_fs)

        assert counters
        assert not any(counter.calls for counter in counters)
        ds.close()
        assert all(counter.calls for counter in counters)

    def test_open_zarrs_through_filesystem(
        self, obsda: xr.DataArray, memory_fs: MemoryFileSystem
    ):
        store = memory_fs.get_mapper("/zarr/test.zarr")
        obsda.to_dataset(name="test").to_zarr(store, zarr_format=2)

        (ds,) = open_zarrs(["/zarr/test.zarr"], filesystem=memory_fs)

        assert np.allclose(ds["test"].values, obsda.values)


class TestStorageOptionsBoundary:
    """``storage_options`` are resolved once, at the driver boundary."""

    def test_merged_into_a_remote_filesystem(self):
        driver = PandasDriver(
            filesystem={"protocol": "memory"},
            options={"storage_options": {"max_paths": 100}, "index_col": 0},
        )
        options = driver.options.get_kwargs()

        fs = resolve_filesystem(driver.filesystem, options)

        assert isinstance(fs, MemoryFileSystem)
        # consumed here, so the reader never sees it
        assert "storage_options" not in options

    def test_left_in_the_options_for_a_local_filesystem(self):
        driver = PandasDriver(options={"storage_options": {"anon": True}})
        options = driver.options.get_kwargs()

        fs = resolve_filesystem(driver.filesystem, options)

        # local paths go straight to the library, which resolves the uri itself
        assert is_local(fs)
        assert options["storage_options"] == {"anon": True}


class TestLocalOnlyReaders:
    def test_open_nc_rejects_remote_uri(self):
        with pytest.raises(ValueError, match="does not support remote paths"):
            open_nc("s3://bucket/test.nc")

    def test_open_geodataset_rejects_remote_uri(self):
        with pytest.raises(ValueError, match="does not support remote paths"):
            open_geodataset("s3://bucket/test.geojson")
