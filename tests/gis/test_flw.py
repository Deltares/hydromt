"""Test hydromt.flw submodule."""

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from hydromt.gis import flw


def test_from_da(flwda):
    # single outlet flow dirs
    flwdir1 = flw.flwdir_from_da(flwda)
    assert np.all(flwdir1.idxs_pit == 6)
    uparea = flwdir1.upstream_area("cell")
    assert uparea.max() == np.multiply(*flwda.shape)
    # test with mask
    mask = uparea > 1
    flwda1 = flwda.assign_coords(mask=(flwda.raster.dims, mask))
    flw1 = flw.flwdir_from_da(flwda1, mask=True)
    flw2 = flw.flwdir_from_da(flwda1, mask=flwda1["mask"])
    assert flw1.upstream_area("cell").max() == mask.sum()
    assert np.all(flw2.idxs_ds == flw1.idxs_ds)
    # errors
    with pytest.raises(TypeError, match="da should be an instance of xarray.DataArray"):
        flw.flwdir_from_da(flwda.values)
    with pytest.raises(ValueError, match="da is missing CRS property"):
        flw.flwdir_from_da(flwda.drop_vars("spatial_ref"))


def test_from_dem(demda, flwdir):
    flwda1 = flw.d8_from_dem(demda, outlets="min")
    # single outlet position
    assert np.all(np.where(flwda1.values.ravel() == 0)[0] == 6)
    # with river shape; fixed depth
    upgrid = flwdir.upstream_area("cell")
    gdf_riv = gpd.GeoDataFrame.from_features(
        flwdir.vectorize(uparea=upgrid, mask=upgrid > 5)
    )
    flwda2 = flw.d8_from_dem(
        demda, gdf_riv=gdf_riv, outlets="min", riv_burn_method="fixed", riv_depth=5
    )
    upg_max = flw.flwdir_from_da(flwda2).upstream_area("cell").max()
    assert upg_max == upgrid.max()
    # with rivdph column
    gdf_riv["rivdph"] = 5
    flwda3 = flw.d8_from_dem(
        demda, gdf_riv=gdf_riv, outlets="min", riv_burn_method="rivdph"
    )
    xr.testing.assert_equal(flwda2, flwda3)
    # with upstream area
    flwda4 = flw.d8_from_dem(demda, gdf_riv=gdf_riv, riv_burn_method="uparea")
    assert np.sum(flwda4 != flwda2) == 21  # 21 cells are different; regression test
    # errors
    with pytest.raises(ValueError, match="uparea column required"):
        flw.d8_from_dem(
            demda, gdf_riv=gdf_riv.drop(columns="uparea"), riv_burn_method="uparea"
        )
    with pytest.raises(ValueError, match="rivdph column required"):
        flw.d8_from_dem(
            demda, gdf_riv=gdf_riv.drop(columns="rivdph"), riv_burn_method="rivdph"
        )
    with pytest.raises(ValueError, match="Unknown riv_burn_method"):
        flw.d8_from_dem(demda, gdf_riv=gdf_riv, riv_burn_method="unknown")
    with pytest.raises(ValueError, match="idxs_pit required"):
        flw.d8_from_dem(demda, gdf_riv=gdf_riv, outlets="idxs_pit")


def test_upscale(hydds, flwdir):
    flwda1, _ = flw.upscale_flwdir(
        hydds, flwdir=flwdir, scale_ratio=3, uparea_name="uparea", method="dmm"
    )
    assert hydds["flwdir"].shape[0] / 3 == flwda1.shape[0]
    assert np.all([v in flwda1.coords for v in ["x_out", "y_out", "idx_out"]])
    # errors
    with pytest.raises(ValueError, match="Flwdir and ds dimensions do not match"):
        flw.upscale_flwdir(
            hydds.isel(x=slice(1, -1)), flwdir=flwdir, scale_ratio=3, method="dmm"
        )


def test_reproject_flwdir(hydds, demda):
    # downscale flwdir
    demda_reproj = demda.raster.reproject(dst_res=demda.raster.res[0] / 2.0)
    hydds["uparea"].data = flw.flwdir_from_da(hydds["flwdir"]).upstream_area("km2")
    hydds1 = flw.reproject_hydrography_like(
        hydds, demda_reproj, outlets="min"
    )  # force single outlet
    assert "uparea" in hydds1
    assert "flwdir" in hydds1
    assert hydds1.raster.shape == demda_reproj.raster.shape
    assert np.allclose(hydds["uparea"].max(), hydds1["uparea"].max())
    # ~ 5% error is acceptable; test also exact value for precise unit testing
    assert abs(1 - hydds["uparea"].max() / hydds1["uparea"].max()) < 0.05
    assert np.isclose(hydds1["uparea"].max(), 1.5)
    # error
    with pytest.raises(ValueError, match="uparea variable not found"):
        flw.reproject_hydrography_like(hydds.drop_vars("uparea"), demda)


def test_basin_map(hydds, flwdir):
    # complete basin
    idx0 = np.argmax(hydds["uparea"].values.ravel())
    da_basins = flw.basin_map(hydds, flwdir, idxs=[idx0])[0]
    da_basins1 = flw.basin_map(hydds, flwdir, outlets=True)[0]
    assert np.all(da_basins == da_basins1)
    # subbasins with stream arguments
    idxs = np.where(hydds["uparea"].values.ravel() == 5)
    da_basins, xy = flw.basin_map(hydds, flwdir, idxs=idxs, uparea=5)
    assert np.all(hydds["uparea"].values[da_basins.values > 0] <= 5)
    # errors
    with pytest.raises(ValueError, match="Flwdir and ds dimensions do not match"):
        flw.basin_map(hydds.isel(x=slice(1, -1)), flwdir)

    flwdir.idxs_outlet = []

    with pytest.raises(ValueError, match="No basin outlets found in domain."):
        flw.basin_map(hydds, flwdir, outlets=True)


def test_clip_basins(hydds, flwdir):
    # clip subbasins
    idx0 = 17
    upa0 = hydds["uparea"].values.flat[idx0]
    hydds_clipped = flw.clip_basins(hydds, flwdir, xy=flwdir.xy(idx0))
    assert hydds_clipped.where(hydds_clipped["mask"]).max() == upa0


def test_gauge_map(hydds, flwdir):
    # test with idxs at outlet
    idx0 = np.argmax(hydds["uparea"].values.ravel())
    da_gauges, idxs, ids = flw.gauge_map(hydds, idxs=[idx0])
    assert idxs[0] == idx0
    assert len(idxs) == 1
    assert np.all(da_gauges.values.flat[idxs] == ids)
    # test with x,y at headwater cell and stream to snap to
    idx1 = np.argmax(flwdir.distnc.ravel())
    stream = hydds["uparea"] >= 10
    xy = hydds.raster.idx_to_xy([idx1])
    da_gauges, idxs, ids = flw.gauge_map(hydds, flwdir=flwdir, stream=stream, xy=xy)
    assert np.all(da_gauges.values.flat[idxs] == ids)
    assert np.all(hydds["uparea"].values.flat[idxs] >= 10)
    # test error
    with pytest.raises(ValueError, match="Either idxs or xy required"):
        flw.gauge_map(hydds)
    # test warning
    with pytest.warns(UserWarning, match="Snapping distance"):
        flw.gauge_map(hydds, flwdir=flwdir, stream=stream, xy=xy, max_dist=0)


def test_outlet_map(hydds, flwdir):
    da_outlets = flw.outlet_map(hydds["flwdir"])
    assert np.all(da_outlets.values.flat[flwdir.idxs_outlet])
    # error
    with pytest.raises(ValueError, match="Unknown pyflwdir ftype"):
        flw.outlet_map(hydds["flwdir"], ftype="unknown")


def test_stream_map(hydds, flwdir):
    hydds["strord"] = xr.DataArray(
        data=flwdir.stream_order(), dims=hydds.raster.dims, coords=hydds.raster.coords
    )
    da_stream = flw.stream_map(hydds, uparea=10, strord=3)
    da_stream1 = np.logical_and(hydds["uparea"] >= 10, hydds["strord"] >= 3)
    assert np.all(da_stream == da_stream1)
    with pytest.raises(ValueError, match="Stream criteria resulted in invalid mask."):
        flw.stream_map(hydds, uparea=hydds["uparea"].max() + 1)


def test_dem_adjust(hydds, demda, flwdir):
    demda1 = flw.dem_adjust(
        demda, hydds["flwdir"], da_rivmsk=hydds["uparea"] > 5, river_d8=True
    )
    assert np.all((demda1.values - flwdir.downstream(demda1.values)) >= 0)
    demda2 = flw.dem_adjust(demda, hydds["flwdir"], flwdir=flwdir, connectivity=8)
    assert np.all((demda2.values - flwdir.downstream(demda2.values)) >= 0)
    with pytest.raises(ValueError, match='Provide "da_rivmsk" in combination'):
        flw.dem_adjust(demda, hydds["flwdir"], river_d8=True)
