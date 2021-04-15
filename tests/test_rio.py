# -*- coding: utf-8 -*-
"""Tests for the hydromt.rio submodule."""

import pytest
import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import box, Point, LineString
from rasterio import features
import dask
import os
import glob
from os.path import join
from pathlib import Path
import rasterio
from rasterio.transform import xy

from hydromt import rio

testdata = [
    ([0.5, 0.0, 3.0, 0.0, -0.5, -9.0], (4, 6)),
    ([0.2, 0.0, 3.0, 0.0, 0.25, -11.0], (8, 15)),
    ([-0.2, 0.0, 6.0, 0.0, 0.5, -11.0], (2, 8, 15)),
]


@pytest.mark.parametrize("transform, shape", testdata)
def test_attrs(transform, shape):
    # checks on rio spatial attributes
    da = rio.full_from_transform(transform, shape, name="test")
    assert isinstance(da.rio.attrs, dict)
    assert rio.GEO_MAP_COORD in da.coords
    assert da.rio.dims == ("y", "x")
    assert "x_dim" in da.rio.attrs
    assert da.rio.dim0 == None if len(shape) == 2 else "dim0"
    assert da.rio.width == da["x"].size
    assert da.rio.height == da["y"].size
    assert da.rio.size == da["x"].size * da["y"].size
    assert np.all(np.isclose(da.rio.res, (transform[0], transform[4])))
    assert da.rio.shape == shape[-2:]
    assert "_FillValue" in da.attrs
    assert np.isnan(da.rio.nodata)


def test_crs():
    da = rio.full_from_transform(*testdata[0], name="test")
    # check crs
    da[rio.GEO_MAP_COORD].attrs = dict()
    da.attrs.update(crs=4326)
    da.to_dataset().rio.set_crs()
    assert da.rio.crs.to_epsg() == 4326
    da[rio.GEO_MAP_COORD].attrs = dict()
    da.attrs.update(crs="unknown", epsg=4326)
    da.rio.set_crs()
    assert da.rio.crs.to_epsg() == 4326
    da[rio.GEO_MAP_COORD].attrs = dict()
    da.rio.set_crs("epsg:4326")
    assert da.rio.crs.is_valid
    assert da.rio.crs.to_epsg() == 4326


def test_attrs_errors(rioda):
    rioda = rioda.rename({"x": "xxxx"})
    with pytest.raises(ValueError, match="dimension not found"):
        rioda.rio.set_spatial_dims()
    rioda = rioda.rename({"xxxx": "x", "y": "yyyy"})
    with pytest.raises(ValueError, match="dimension not found"):
        rioda.rio.set_spatial_dims()
    rioda = rioda.rename({"yyyy": "y"})
    rioda["x"] = np.random.rand(rioda["x"].size)
    with pytest.raises(ValueError, match="only applies to regular grids"):
        rioda.rio.set_spatial_dims()
    with pytest.raises(ValueError, match="Invalid dimension order."):
        rioda.transpose("x", "y").rio._check_dimensions()
    with pytest.raises(ValueError, match="Invalid dimension order."):
        da1 = rioda.expand_dims("t").transpose("y", "x", "t", transpose_coords=True)
        da1.rio._check_dimensions()
    with pytest.raises(ValueError, match="Only 2D and 3D data"):
        rioda.expand_dims(("t", "t1")).rio._check_dimensions()


def test_from_numpy_full_like():
    da = rio.full_from_transform(*testdata[0], nodata=-1, name="test")
    da0 = rio.full_like(da)
    da1 = rio.RasterDataArray.from_numpy(
        da.values[None, :], da.rio.transform, da.rio.nodata, da.attrs, da.rio.crs
    )
    assert np.all(da == da0)
    assert np.all(da == da1)
    assert da.attrs == da1.attrs
    assert da.rio.shape == da1.rio.shape
    assert np.all(da.rio.transform == da1.rio.transform)
    assert da.rio.crs == da1.rio.crs
    with pytest.raises(ValueError, match="Only 2D and 3D"):
        rio.RasterDataArray.from_numpy(da.values[None, None, :], da.rio.transform)
    with pytest.raises(ValueError, match="Only 2D and 3D"):
        rio.full_from_transform(da.rio.transform, (1, 1, 5, 5))
    ds1 = rio.RasterDataset.from_numpy(
        data_vars={"var0": da.values, "var1": (da.values, da.rio.nodata)},
        transform=da.rio.transform,
        crs=da.rio.crs,
    )
    assert ds1["var1"].rio.nodata == da.rio.nodata
    assert da.rio.crs == ds1.rio.crs
    with pytest.raises(xr.core.merge.MergeError, match="Data shapes do not match"):
        ds1 = rio.RasterDataset.from_numpy(
            data_vars={"var0": da.values, "var1": da.values.T},
            transform=da.rio.transform,
        )
    with pytest.raises(ValueError, match="should be xarray.DataArray"):
        rio.full_like(da.to_dataset())


@pytest.mark.parametrize("transform, shape", testdata)
def test_idx(transform, shape):
    da = rio.full_from_transform(transform, shape, name="test")
    size = np.multiply(*da.rio.shape)
    row, col = da.rio.height - 1, da.rio.width - 1
    assert xy(da.rio.transform, 0, 0) == da.rio.idx_to_xy(0)
    assert xy(da.rio.transform, row, col) == da.rio.idx_to_xy(size - 1)
    assert np.all(np.isnan(da.rio.idx_to_xy(size, mask_outside=True)))
    with pytest.raises(ValueError, match="outside domain"):
        da.rio.idx_to_xy(size)
    assert da.rio.xy_to_idx(da.rio.xcoords[0], da.rio.ycoords[0]) == 0
    assert da.rio.xy_to_idx(da.rio.xcoords[-1], da.rio.ycoords[-1]) == size - 1
    assert np.all(da.rio.xy_to_idx(-999, -999, mask_outside=True) == -1)
    with pytest.raises(ValueError, match="outside domain"):
        da.rio.xy_to_idx(-999, -999)


def test_rasterize(rioda):
    gdf = gpd.GeoDataFrame(geometry=[box(*rioda.rio.internal_bounds)])
    gdf["id"] = [3]
    assert np.all(rioda.rio.rasterize(gdf, col_name="id") == 3)
    assert np.all(rioda.rio.rasterize(gdf, col_name="id", sindex=True) == 3)
    mask = rioda.rio.geometry_mask(gdf)
    assert mask.dtype == np.bool
    assert np.all(mask)
    with pytest.raises(ValueError, match="No shapes found"):
        rioda.rio.rasterize(gpd.GeoDataFrame())


def test_vectorize():
    da = rio.full_from_transform(*testdata[0], nodata=1, name="test")
    da.rio.set_crs(4326)
    # all nodata
    gdf = da.rio.vectorize()
    assert gdf.index.size == 0
    # edit nodata > all value 1
    da.rio.set_nodata(np.nan)
    gdf = da.rio.vectorize()
    assert np.all(gdf["value"].values == 1)
    assert da.rio.crs.to_epsg() == gdf.crs.to_epsg()
    assert np.all(da == da.rio.geometry_mask(gdf).astype(da.dtype))


@pytest.mark.parametrize("transform, shape", testdata)
def test_clip(transform, shape):
    # create rasterdataarray with crs
    da = rio.full_from_transform(transform, shape, name="test", crs=4326)
    # create gdf covering half raster
    w, s, _, n = da.rio.bounds
    i = int(round(da.rio.shape[-1] / 2))
    e = da.rio.xcoords.values[i] + da.rio.res[0] / 2.0
    gdf = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs=da.rio.crs)
    # test bbox
    da_clip0 = da.rio.clip_bbox(gdf.total_bounds)
    assert np.all(np.isclose(da_clip0.rio.bounds, gdf.total_bounds))
    # test bbox - buffer
    da_clip = da.rio.clip_bbox(gdf.total_bounds, buffer=da.rio.width)
    assert np.all(np.isclose(da.rio.bounds, da_clip.rio.bounds))
    # test bbox - align
    align = np.round(abs(da.rio.res[0] * 2), 2)
    da_clip = da.rio.clip_bbox(gdf.total_bounds, align=align)
    dalign = np.round(da_clip.rio.bounds[2], 2) % align
    assert np.isclose(dalign, 0) or np.isclose(dalign, align)
    # test geom
    da_clip1 = da.rio.clip_geom(gdf)
    assert np.all(np.isclose(da_clip1.rio.bounds, da_clip0.rio.bounds))
    # test geom - different crs
    da_clip1 = da.rio.clip_geom(gdf.to_crs(3857))
    assert np.all(np.isclose(da_clip1.rio.bounds, da_clip0.rio.bounds))
    # test mask
    da_clip1 = da.rio.clip_mask(da.rio.geometry_mask(gdf))
    assert np.all(np.isclose(da_clip1.rio.bounds, da_clip0.rio.bounds))


def test_clip_errors(rioda):
    with pytest.raises(ValueError, match="Mask should be xarray.DataArray type."):
        rioda.rio.clip_mask(rioda.values)
    with pytest.raises(ValueError, match="Mask shape invalid."):
        rioda.rio.clip_mask(rioda.isel({"x": slice(1, -1)}))
    with pytest.raises(ValueError, match="Invalid mask."):
        rioda.rio.clip_mask(xr.zeros_like(rioda))
    with pytest.raises(ValueError, match="should be geopandas"):
        rioda.rio.clip_geom(rioda.rio.bounds)


def test_reproject():
    kwargs = dict(name="test", crs=4326)
    transform, shape = testdata[1][0], (9, 5, 5)
    da0 = rio.full_from_transform(transform, shape, **kwargs)
    da0.data = np.random.random(da0.shape)
    ds0 = da0.to_dataset()
    ds1 = rio.full_from_transform(*testdata[1], **kwargs).to_dataset()
    assert np.all(ds1.rio.bounds == ds1.rio.transform_bounds(ds1.rio.crs))
    ds2 = ds1.rio.reproject(dst_crs=3857, method="nearest_index")
    assert ds2.rio.crs.to_epsg() == 3857
    ds2 = ds1.rio.reproject(dst_crs=3857, dst_res=1000, align=True)
    assert np.all(np.asarray(ds2.rio.bounds)) // 1000 == 0
    ds2 = ds1.rio.reproject(dst_width=4, dst_height=2, method="average")
    assert np.all(ds2.rio.shape == (2, 4))
    ds2 = ds1.rio.reproject_like(ds0)
    assert np.all(ds2.rio.xcoords == ds0.rio.xcoords)
    ds2 = da0.rio.reproject_like(ds1, method="nearest_index")
    assert np.all(ds2.rio.xcoords == ds0.rio.xcoords)
    ds2 = ds1.rio.reproject(dst_crs="utm")
    assert ds2.rio.crs.is_projected
    index = ds1.rio.nearest_index(dst_crs="utm")
    assert isinstance(index, xr.DataArray)
    assert np.all(index.values >= -1)  ## -1 for invalid indices (outside domain)
    ds2_index = ds1.rio.reindex2d(index)
    assert np.all([np.all(ds2_index[c] == ds2[c]) for c in ds2_index.coords])
    # test with chunks
    da2_lazy = da0.chunk({da0.rio.dim0: 3}).rio.reproject(dst_crs="utm")
    assert isinstance(da2_lazy.data, dask.array.core.Array)
    assert np.all(ds2["test"] == da2_lazy.compute())
    da2_lazy = da0.chunk({da0.rio.dim0: 3}).rio.reproject(
        dst_crs="utm", method="nearest_index"
    )
    assert isinstance(da2_lazy.data, dask.array.core.Array)
    assert np.all(ds2_index["test"] == da2_lazy.compute())
    # check error messages
    with pytest.raises(ValueError, match="Resampling method unknown"):
        ds1.rio.reproject(dst_crs=3857, method="unknown")
    ds1.rio.set_attrs(crs_wkt=None)
    with pytest.raises(ValueError, match="CRS is missing"):
        ds1.rio.reproject(dst_crs=3857)


def test_interpolate_na():
    # mv > nan
    da0 = rio.full_from_transform(*testdata[0], nodata=-1)
    da0.values[0, 0] = 1
    da1 = da0.rio.mask_nodata().rio.interpolate_na()
    assert np.all(np.isnan(da1) == False)
    assert np.all(da1 == 1)
    assert np.all(np.isnan(da1.rio.interpolate_na()) == False)
    assert np.all(da0.rio.interpolate_na() != da0.rio.nodata)
    assert np.all(da0.expand_dims("t").rio.interpolate_na() != da0.rio.nodata)
    da2 = da0.astype(np.int32)  # this removes the nodata value ...
    da2.rio.set_nodata(-9999)
    assert da2.rio.interpolate_na().dtype == np.int32


def test_vector_grid(rioda):
    gdf = rioda.rio.vector_grid()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert rioda.rio.size == gdf.index.size


def test_sample():
    transform, shape = [0.2, 0.0, 3.0, 0.0, 0.25, -11.0], (8, 10)
    da = rio.full_from_transform(transform, shape, name="test", crs=4326)
    da.data = np.arange(da.rio.size).reshape(da.rio.shape)
    gdf0 = gpd.GeoDataFrame(geometry=gpd.points_from_xy(da.x.values[:8], da.y.values))

    values = np.arange(0, 78, 11)
    assert np.all(da.to_dataset().rio.sample(gdf0)["test"] == values)

    nneighbors = np.array([4, 9, 9, 9, 9, 9, 9, 6])
    assert np.all(da.rio.sample(gdf0, wdw=1).count("wdw") == nneighbors)
    assert np.all(da.rio.sample(gdf0, wdw=1).median("wdw").values[1:-1] == values[1:-1])

    with pytest.raises(ValueError, match="Only point geometries accepted"):
        da.rio.sample(gdf0.buffer(1))


def test_zonal_stats():
    transform, shape = [0.2, 0.0, 3.0, 0.0, -0.25, -11.0], (8, 10)
    da = rio.full_from_transform(transform, shape, name="test", crs=4326)
    da.data = np.arange(da.rio.size).reshape(da.rio.shape)
    ds = xr.merge([da, da.expand_dims("time").to_dataset().rename({"test": "test1"})])
    w, s, e, n = da.rio.bounds
    geoms = [
        box(w, s, w + abs(e - w) / 2.0, n),
        box(w - 2, s, w - 0.2, n),  # outside
        Point((w, n)),
        LineString([(w, n), (e, s)]),
    ]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=da.rio.crs)

    ds0 = da.rio.zonal_stats(gdf, "count")
    assert "test_count" in ds0.data_vars and "index" in ds0.dims
    assert np.all(ds0["index"] == np.array([0, 2, 3]))
    assert np.all(ds0["test_count"] == np.array([40, 1, 10]))

    ds0 = ds.rio.zonal_stats(gdf.to_crs(3857), [np.nanmean, "mean"])
    ds1 = ds.rio.zonal_stats(gdf, [np.nanmean, "mean"])
    assert np.all(ds0["test_nanmean"] == ds0["test_mean"])
    assert np.all(ds1["test_mean"] == ds0["test_mean"])

    with pytest.raises(ValueError, match="Stat asdf not valid"):
        da.rio.zonal_stats(gdf, "asdf")
    with pytest.raises(IndexError, match="All geometries outside raster domain"):
        da.rio.zonal_stats(gdf.iloc[1:2], "mean")
