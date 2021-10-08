"""Test for hydromt.gis_utils submodule"""

import pytest
import numpy as np
from hydromt import gis_utils as gu
from hydromt.raster import full_from_transform, RasterDataArray
from rasterio.transform import from_origin


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
    xs, ys = gu.affine_to_coords(transform, shape)
    assert np.all(ys == 90 - np.arange(0.5, shape[0]))
    assert np.all(xs == np.arange(0.5, shape[1]))

    # offset for geographic crs
    da = full_from_transform(transform, shape, crs=4326)
    da1 = gu.meridian_offset(da, x_name="x")
    assert da1.raster.bounds[0] == -180
    da2 = gu.meridian_offset(da1, x_name="x", bbox=[170, 0, 190, 10])
    assert da2.raster.bounds[0] == 170
    da3 = gu.meridian_offset(da1, x_name="x", bbox=[-190, 0, -170, 10])
    assert da3.raster.bounds[2] == -170


def test_area_res():
    # surface area of earth should be approx 510.100.000 km2
    transform = from_origin(-180, 90, 1, 1)
    shape = (180, 360)
    da = full_from_transform(transform, shape, crs=4326)
    assert np.isclose(da.raster.area_grid().sum() / 1e6, 510064511.156224)
    assert gu.cellres(0) == (111319.458, 110574.2727)


def test_gdf(world):
    assert np.all(
        gu.filter_gdf(
            world,
            world.iloc[
                [0],
            ].to_crs(3857),
        )
        == 0
    )
    assert (
        world.iloc[
            gu.filter_gdf(world, bbox=[3, 51.5, 4, 52])[0],
        ]["iso_a3"]
        == "NLD"
    )
    with pytest.raises(ValueError, match="Unknown geometry mask type"):
        gu.filter_gdf(world, geom=[3, 51.5, 4, 52])


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
