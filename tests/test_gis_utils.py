"""Test for hydromt.gis_utils submodule"""

import pytest
import numpy as np
from hydromt import gis_utils as gu
from hydromt.raster import full_from_transform, RasterDataArray
from rasterio.transform import from_origin
from affine import Affine


def test_crs():
    bbox = [3, 51.5, 4, 52]  # NL
    assert gu.utm_crs(bbox).to_epsg() == 32631
    assert gu.parse_crs("utm", bbox).to_epsg() == 32631
    bbox1 = [-77.5, -12.2, -77.0, -12.0]
    assert gu.utm_crs(bbox1).to_epsg() == 32718
    _, _, xattrs, yattrs = gu.axes_attrs(gu.parse_crs(4326))
    assert xattrs["units"] == "degrees_east"
    assert yattrs["units"] == "degrees_north"
    _, _, xattrs, yattrs = gu.axes_attrs(gu.utm_crs(bbox1))
    assert xattrs["units"] == yattrs["units"] == "m"


def test_transform():
    transform = from_origin(0, 90, 1, 1)
    shape = (180, 360)
    coords = gu.affine_to_coords(transform, shape)
    xs, ys = coords["x"][1], coords["y"][1]
    assert np.all(ys == 90 - np.arange(0.5, shape[0]))
    assert np.all(xs == np.arange(0.5, shape[1]))

    # offset for geographic crs
    da = full_from_transform(transform, shape, crs=4326)
    assert np.allclose(da.raster.origin, np.array([0, 90]))
    da1 = gu.meridian_offset(da, x_name="x")
    assert da1.raster.bounds[0] == -180
    da2 = gu.meridian_offset(da1, x_name="x", bbox=[170, 0, 190, 10])
    assert da2.raster.bounds[0] == 0
    da3 = gu.meridian_offset(da1, x_name="x", bbox=[-190, 0, -170, 10])
    assert da3.raster.bounds[2] == 0


def test_transform_rotation():
    # with rotation
    transform = Affine.rotation(30) * Affine.scale(1, 2)
    shape = (10, 5)
    coords = gu.affine_to_coords(transform, shape)
    xs, ys = coords["xc"][1], coords["yc"][1]
    assert xs.ndim == 2 and ys.ndim == 2
    da = full_from_transform(transform, shape, crs=4326)
    assert da.raster.x_dim == "x"
    assert da.raster.xcoords.ndim == 2
    assert np.allclose(transform, da.raster.transform)


def test_area_res():
    # surface area of earth should be approx 510.100.000 km2
    transform = from_origin(-180, 90, 1, 1)
    shape = (180, 360)
    da = full_from_transform(transform, shape, crs=4326)
    assert np.isclose(da.raster.area_grid().sum() / 1e6, 510064511.156224)
    assert gu.cellres(0) == (111319.458, 110574.2727)


def test_gdf(world):
    country = world.iloc[[0], :].to_crs(3857)
    assert np.all(gu.filter_gdf(world, country) == 0)
    idx0 = gu.filter_gdf(world, bbox=[3, 51.5, 4, 52])[0]
    assert (
        world.iloc[
            idx0,
        ]["iso_a3"]
        == "NLD"
    )
    with pytest.raises(ValueError, match="Unknown geometry mask type"):
        gu.filter_gdf(world, geom=[3, 51.5, 4, 52])


def test_nearest(world, geodf):
    idx, _ = gu.nearest(geodf, geodf)
    assert np.all(idx == geodf.index)
    idx, dst = gu.nearest(geodf, world)
    assert np.all(dst == 0)
    assert np.all(world.loc[idx, "name"].values == geodf["country"].values)
    gdf0 = geodf.copy()
    gdf0["iso_a3"] = ""
    gdf1 = gu.nearest_merge(geodf, world.drop(idx), max_dist=1e6)
    assert np.all(gdf1.loc[gdf1["distance_right"] > 1e6, "index_right"] == -1)
    assert np.all(gdf1.loc[gdf1["distance_right"] > 1e6, "iso_a3"] != "")


def test_spread():
    transform = from_origin(-15, 10, 1, 1)
    shape = (20, 30)
    data = np.zeros(shape)
    data[10, 10] = 1  # lin index 310
    frc = np.ones(shape)
    msk = np.ones(shape, dtype=bool)
    da_obs = RasterDataArray.from_numpy(data, transform=transform, nodata=0, crs=4326)
    da_msk = RasterDataArray.from_numpy(msk, transform=transform, crs=4326)
    da_frc = RasterDataArray.from_numpy(frc, transform=transform, crs=4326)
    # only testing the wrapping of pyflwdir method, not the method itself
    ds_out = gu.spread2d(da_obs, da_friction=da_frc, da_mask=da_msk)
    assert np.all(ds_out["source_value"] == 1)
    assert np.all(ds_out["source_idx"] == 310)
    assert ds_out["source_dst"].values[10, 10] == 0
    with pytest.raises(ValueError, match='"nodata" must be a finite value'):
        gu.spread2d(da_obs, nodata=np.nan)
