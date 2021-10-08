"""Test hydromt.flw submodule"""
import pytest
import numpy as np
import xarray as xr
from hydromt import flw
import geopandas as gpd


def test_from_da(flwda, flwdir):
    flwdir1 = flw.flwdir_from_da(
        flwda,
    )
    assert np.all(flwdir1.idxs_ds == flwdir.idxs_ds)


def test_from_dem(flwda, demda, flwdir):
    flwda1 = flw.d8_from_dem(demda, outlets="min")
    assert np.all(flwda.values == flwda1.values)
    # with river shape (simplistic example)
    upgrid = flwdir.upstream_area("cell")
    gdf_stream = gpd.GeoDataFrame.from_features(
        flwdir.vectorize(uparea=upgrid, mask=upgrid > 5)
    )
    flwda2 = flw.d8_from_dem(demda, gdf_stream=gdf_stream, outlets="min")
    upg_max = flw.flwdir_from_da(flwda2).upstream_area("cell").max()
    assert upg_max == upgrid.max()


def test_upscale(hydds, flwdir):
    flwda1, _ = flw.upscale_flwdir(hydds, flwdir, 3, uparea_name="uparea", method="dmm")
    assert hydds["flwdir"].shape[0] / 3 == flwda1.shape[0]
    assert np.all([v in flwda1.coords for v in ["x_out", "y_out", "idx_out"]])


def test_reproject_flwdir(hydds, demda):
    demda_reproj = demda.raster.reproject(dst_crs=4326)
    hydds1 = flw.reproject_hydrography_like(hydds, demda_reproj)
    assert hydds1.raster.crs == demda_reproj.raster.crs
    assert "uparea" in hydds1 and "flwdir" in hydds1


def test_basin_map(hydds, flwdir):
    idx0 = np.argmax(hydds["uparea"].values.ravel())
    assert np.all(flw.basin_map(hydds, flwdir, idxs=[idx0], uparea=5)[0].values == 1)
