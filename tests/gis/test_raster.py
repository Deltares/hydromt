# -*- coding: utf-8 -*-
"""Tests for the hydromt.raster submodule."""

import os
from os.path import isfile, join
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from affine import Affine
from shapely.geometry import LineString, Point, box

from hydromt.gis import gis_utils, raster
from hydromt.io import open_raster

# origin, rotation, res, shape, internal_bounds
# NOTE a rotated grid with a negative dx is not supported
tests = [
    ((3, -9), 0, (0.5, -0.5), (4, 6), (3, -9, 6, -11)),
    ((3, -11), 0, (0.2, 0.25), (8, 15), (3, -11, 6, -9)),
    ((6, -11), 0, (-0.2, 0.5), (2, 8, 15), (6, -11, 3, -7)),
    ((0, 0), 30, (2, 1), (12, 8), (-6, 0, 13.856406460551018, 18.39230484541326)),
    (
        (-10, 3),
        170,
        (2, -1),
        (7, 8),
        (-25.75692404819533, 12.672025113756337, -8.78446275633149, 3.0),
    ),
    ((-2, -3), 270, (2, -5), (3, 2), (-17.0, -3.0, -2.0, -7.0)),
]


def get_transform(
    origin: tuple[float, float], rotation: float, res: tuple[float, float]
) -> Affine:
    return Affine.translation(*origin) * Affine.rotation(rotation) * Affine.scale(*res)


testdata = [(get_transform(*d[:3]), d[-2]) for d in tests[:4]]


@pytest.mark.parametrize(("origin", "rotation", "res", "shape", "bounds"), tests)
def test_raster_properties(origin, rotation, res, shape, bounds):
    transform = get_transform(origin, rotation, res)
    da = raster.full_from_transform(transform, shape, name="test", crs=4326)
    assert np.allclose(rotation, da.raster.rotation)
    assert np.allclose(res, da.raster.res)
    assert np.allclose(origin, da.raster.origin)
    assert np.allclose(transform, da.raster.transform)
    assert np.allclose(da.raster.box.total_bounds, da.raster.bounds)
    assert np.allclose(bounds, da.raster.internal_bounds)
    assert da.raster.box.crs == da.raster.crs
    # attributes do not persist after slicing
    assert da[:2, :2].raster._transform is None
    assert da[:2, :2].raster._crs is None


@pytest.mark.parametrize(("transform", "shape"), testdata)
def test_attrs(transform, shape):
    # checks on raster spatial attributes
    da = raster.full_from_transform(transform, shape, name="test")
    da.drop_vars(raster.GEO_MAP_COORD)  # reset attrs
    assert isinstance(da.raster.attrs, dict)
    assert raster.GEO_MAP_COORD in da.coords
    assert da.raster.dims == ("y", "x")
    assert "x_dim" in da.raster.attrs
    assert da.raster.dim0 is None if len(shape) == 2 else "dim0"
    assert da.raster.width == da["x"].size
    assert da.raster.height == da["y"].size
    assert da.raster.size == da["x"].size * da["y"].size
    assert da.raster.shape == shape[-2:]
    assert "_FillValue" in da.attrs
    assert np.isnan(da.raster.nodata)


def test_crs():
    da = raster.full_from_transform(*testdata[0], name="test")
    # check crs
    da[raster.GEO_MAP_COORD].attrs = dict()
    da.attrs.update(crs=4326)
    ds = da.to_dataset()
    ds.raster.set_crs()
    assert ds.raster.crs.to_epsg() == 4326
    da[raster.GEO_MAP_COORD].attrs = dict()
    da.attrs.update(crs="unknown", epsg=4326)
    da.raster.set_crs()
    assert da.raster.crs.to_epsg() == 4326
    da[raster.GEO_MAP_COORD].attrs = dict()
    da.raster.set_crs("epsg:4326")
    assert da.raster.crs.to_epsg() == 4326
    # compound crs
    da[raster.GEO_MAP_COORD].attrs = dict()
    da.raster.set_crs(9518)  # WGS 84 + EGM2008 height
    assert da.raster.crs.to_epsg() == 9518  # return horizontal crs


def test_gdal(tmpdir):
    da = raster.full_from_transform(*testdata[0], name="test")
    # Add crs
    da.attrs.update(crs=4326)
    da.raster.set_crs()
    # Update gdal compliant attrs
    da1 = da.raster.gdal_compliant(rename_dims=True, force_sn=True)
    assert raster.GEO_MAP_COORD in da1.coords
    assert da1.raster.dims == ("latitude", "longitude")
    assert da1.raster.res[1] > 0
    # Write to netcdf and reopen with gdal
    netcdf_path = str(tmpdir.join("gdal_test.nc"))
    da1.to_netcdf(netcdf_path)
    with rasterio.open(netcdf_path) as src:
        src.read()
        assert da.raster.crs == src.crs
    # # test with web mercator dataset
    da2 = da.raster.reproject(dst_crs=3857)
    ds2 = da2.to_dataset().raster.gdal_compliant(rename_dims=True, force_sn=True)
    netcdf_path = str(tmpdir.join("gdal_test_epsg3857.nc"))
    ds2.to_netcdf(netcdf_path)
    with rasterio.open(netcdf_path) as src:
        src.read()
        assert ds2.raster.crs == src.crs


def test_attrs_errors(rioda):
    rioda = rioda.rename({"x": "X"})
    rioda.raster.set_spatial_dims()
    assert rioda.raster.x_dim == "X"
    rioda = rioda.rename({"X": "xxxx"})
    with pytest.raises(ValueError, match="dimension not found"):
        rioda.raster.set_spatial_dims()
    rioda = rioda.rename({"xxxx": "x", "y": "yyyy"})
    with pytest.raises(ValueError, match="dimension not found"):
        rioda.raster.set_spatial_dims()
    rioda = rioda.rename({"yyyy": "y"})
    with pytest.raises(ValueError, match="Invalid dimension order."):
        rioda.transpose("x", "y").raster._check_dimensions()

    da1 = rioda.expand_dims("t").transpose("y", "x", "t", transpose_coords=True)
    with pytest.raises(ValueError, match="Invalid dimension order."):
        da1.raster._check_dimensions()
    with pytest.raises(ValueError, match="Only 2D and 3D data"):
        rioda.expand_dims(("t", "t1")).raster._check_dimensions()


def test_check_dimensions(rioda):
    # test with 2D data
    rioda.raster._check_dimensions()
    assert "dim0" not in rioda.raster.attrs.keys()
    # test with 3D data
    rioda_3d = rioda.expand_dims("t")
    rioda_3d.name = "test_3d"
    rioda_3d.raster._check_dimensions()
    assert rioda_3d.raster.attrs["dim0"] == "t"
    # test with dataset of 2D and 3D data
    ds = xr.merge([rioda, rioda_3d])
    ds.raster._check_dimensions()
    assert ds.raster.attrs["dim0"] == "t"
    # add 3D data with a different dimension name
    rioda_3d_t1 = rioda.expand_dims("t1")
    rioda_3d_t1.name = "test_3d_t1"
    ds = xr.merge([ds, rioda_3d, rioda_3d_t1])
    ds.raster._check_dimensions()
    assert "dim0" not in ds.raster.attrs.keys()


def test_from_numpy_full_like():
    # test full with rotated grid
    da_rot = raster.full_from_transform(*testdata[-1], nodata=-1, name="test")
    da_rot1 = raster.full(da_rot.raster.coords, nodata=-1, name="test")
    assert da_rot1.raster.identical_grid(da_rot1)
    # test with normal grid
    da = raster.full_from_transform(*testdata[0], nodata=-1, name="test")
    da0 = raster.full_like(da)
    da1 = raster.RasterDataArray.from_numpy(
        da.values[None, :],
        da.raster.transform,
        da.raster.nodata,
        da.attrs,
        da.raster.crs,
    )
    assert np.all(da == da0)
    assert np.all(da == da1)
    assert da.attrs == da1.attrs
    assert da.raster.shape == da1.raster.shape
    assert np.all(da.raster.transform == da1.raster.transform)
    assert da.raster.crs == da1.raster.crs
    with pytest.raises(ValueError, match="Only 2D and 3D"):
        raster.RasterDataArray.from_numpy(da.values[None, None, :], da.raster.transform)
    with pytest.raises(ValueError, match="Only 2D and 3D"):
        raster.full_from_transform(da.raster.transform, (1, 1, 5, 5))
    ds1 = raster.RasterDataset.from_numpy(
        data_vars={"var0": da.values, "var1": (da.values, da.raster.nodata)},
        transform=da.raster.transform,
        crs=da.raster.crs,
    )
    assert ds1["var1"].raster.nodata == da.raster.nodata
    assert da.raster.crs == ds1.raster.crs
    with pytest.raises(xr.core.merge.MergeError, match="Data shapes do not match"):
        ds1 = raster.RasterDataset.from_numpy(
            data_vars={"var0": da.values, "var1": da.values.T},
            transform=da.raster.transform,
        )
    with pytest.raises(ValueError, match="should be xarray.DataArray"):
        raster.full_like(da.to_dataset())


@pytest.mark.parametrize(("transform", "shape"), testdata)
def test_idx(transform, shape):
    da = raster.full_from_transform(transform, shape, name="test")
    size = np.multiply(*da.raster.shape)
    xs, ys = da.raster.xcoords.values.ravel(), da.raster.ycoords.values.ravel()
    assert np.allclose(([xs[0]], [ys[0]]), da.raster.idx_to_xy(0))
    assert np.allclose(([xs[-1]], [ys[-1]]), da.raster.idx_to_xy(size - 1))
    assert np.all(np.isnan(da.raster.idx_to_xy(size, mask_outside=True)))
    with pytest.raises(ValueError, match="outside domain"):
        da.raster.idx_to_xy(size)
    assert da.raster.xy_to_idx(xs[0], ys[0]) == 0
    assert da.raster.xy_to_idx(xs[-1], ys[-1]) == size - 1
    assert np.all(da.raster.xy_to_idx(-999, -999, mask_outside=True) == -1)
    with pytest.raises(ValueError, match="outside domain"):
        da.raster.xy_to_idx(-999, -999)


def test_rasterize(rioda):
    gdf = gpd.GeoDataFrame(geometry=[box(*rioda.raster.internal_bounds)])
    gdf["id"] = [3]
    assert np.all(rioda.raster.rasterize(gdf, col_name="id") == 3)
    assert np.all(rioda.raster.rasterize(gdf, col_name="id", sindex=True) == 3)
    mask = rioda.raster.geometry_mask(gdf)
    assert mask.dtype == bool
    assert np.all(mask)
    with pytest.raises(ValueError, match="No shapes found"):
        rioda.raster.rasterize(
            gdf=gpd.GeoDataFrame(geometry=[box(-5, 2, -3, 4)], crs=rioda.raster.crs),
            sindex=True,
        )


def test_rasterize_geometry(rioda):
    xmin, ymin, xmax, ymax = rioda.raster.bounds
    resx, resy = rioda.raster.res
    box1 = box(xmin, ymin, xmin + resx, ymin - resy)
    box2 = box(xmax - resx, ymax + resy, xmax, ymax)
    x1 = rioda.raster.xcoords.values[2]
    y1 = rioda.raster.ycoords.values[2]
    box3 = box(x1, y1, x1 + resx, y1 - resy)
    gdf = gpd.GeoDataFrame(geometry=[box1, box2, box3], crs=rioda.raster.crs)

    da = rioda.raster.rasterize_geometry(
        gdf, method="fraction", name="frac", nodata=0, keep_geom_type=False
    )
    assert da.name == "frac"
    assert da.raster.nodata == 0
    assert np.round(da.values.max(), 4) <= 1.0
    # Count unique values in da
    assert np.unique(np.round(da.values, 2)).size == 3
    assert 0.25 in np.unique(np.round(da.values, 2))

    da2 = rioda.raster.rasterize_geometry(gdf, method="area")
    assert da2.name == "area"
    assert da2.raster.nodata == -1.0
    rioda_grid = rioda.raster.vector_grid()
    crs_utm = gis_utils.parse_crs("utm", rioda_grid.total_bounds)
    rioda_grid = rioda_grid.to_crs(crs_utm)
    assert da2.values.max() == rioda_grid.area.max()


def test_vectorize():
    da = raster.full_from_transform(*testdata[0], nodata=1, name="test")
    da.raster.set_crs(4326)
    # all nodata
    gdf = da.raster.vectorize()
    assert gdf.index.size == 0
    # edit nodata > all value 1
    da.raster.set_nodata(np.nan)
    gdf = da.raster.vectorize()
    assert np.all(gdf["value"].values == 1)
    assert da.raster.crs.to_epsg() == gdf.crs.to_epsg()
    assert np.all(da == da.raster.geometry_mask(gdf).astype(da.dtype))
    # test with compound crs
    da.raster.set_crs(9518)  # WGS 84 + EGM2008 height
    gdf = da.raster.vectorize()
    assert gdf.crs.to_epsg() == 9518


@pytest.mark.parametrize(("transform", "shape"), testdata)
def test_clip(transform, shape):
    # create rasterdataarray with crs
    da = raster.full_from_transform(transform, shape, nodata=1, name="test", crs=4326)
    da.raster.set_nodata(0)
    # attributes do not persist with xarray slicing
    da1 = da[:2, :2]
    assert da1.raster._transform is None
    assert da1.raster._crs is None
    # attributes do persisit when slicing using clip_bbox
    raster1d = da.raster.clip(slice(1, 2), slice(0, 2))
    assert raster1d.raster._crs is not None
    assert raster1d.raster.transform is not None
    # create gdf covering approx half raster
    w, s, e, n = da.raster.bounds
    gdf = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs=da.raster.crs)
    # test bbox - buffer
    da_clip = da.raster.clip_bbox(gdf.total_bounds, buffer=da.raster.width)
    assert np.all(np.isclose(da.raster.bounds, da_clip.raster.bounds))
    # test bbox - no overlap with raster
    # left
    da_clip = da.raster.clip_bbox((w - 0.2, s, w - 0.1, n))
    assert len(da_clip[da_clip.raster.x_dim]) == 0
    # right
    da_clip = da.raster.clip_bbox((e + 0.1, s, e + 0.2, n))
    assert len(da_clip[da_clip.raster.x_dim]) == 0
    # top
    da_clip = da.raster.clip_bbox((w, n + 0.1, e, n + 0.2))
    assert len(da_clip[da_clip.raster.y_dim]) == 0
    # bottom
    da_clip = da.raster.clip_bbox((w, s - 0.2, e, s - 0.1))
    assert len(da_clip[da_clip.raster.y_dim]) == 0
    # test bbox
    da_clip0 = da.raster.clip_bbox(gdf.total_bounds)
    # test geom
    da_clip1 = da.raster.clip_geom(gdf)
    assert np.all(np.isclose(da_clip1.raster.bounds, da_clip0.raster.bounds))
    assert "mask" not in da_clip1.coords  # this changed in v0.7.2
    # test mask
    da_mask = da.raster.geometry_mask(gdf)
    da_clip1 = da.raster.clip_mask(da_mask=da_mask)
    assert np.all(np.isclose(da_clip1.raster.bounds, da_clip0.raster.bounds))
    assert "mask" not in da_clip1.coords  # this changed in v0.7.2
    da_clip1 = da.raster.clip_mask(da_mask=da_mask, mask=True)
    assert "mask" in da_clip1.coords

    # test geom - different crs & mask=True (changed in v0.7.2)
    da_clip1 = da.raster.clip_geom(gdf.to_crs(3857), mask=True)
    assert np.all(np.isclose(da_clip1.raster.bounds, da_clip0.raster.bounds))
    assert "mask" in da_clip1.coords

    # these test are for non-rotated only
    if da.raster.rotation != 0:
        return
    assert np.all(np.isclose(da_clip0.raster.bounds, gdf.total_bounds))


def test_clip_align(rioda):
    # test align
    bbox = (3.5, -10.5, 5.5, -9.5)
    da_clip = rioda.raster.clip_bbox(bbox)
    assert np.all(np.isclose(da_clip.raster.bounds, bbox))
    da_clip = rioda.raster.clip_bbox(bbox, align=1)
    assert da_clip.raster.bounds == (3, -11, 6, -9)


def test_clip_errors(rioda):
    with pytest.raises(ValueError, match="Mask should be xarray.DataArray type."):
        rioda.raster.clip_mask(rioda.values)
    with pytest.raises(ValueError, match="Mask grid invalid"):
        rioda.raster.clip_mask(rioda.isel({"x": slice(1, -1)}))
    with pytest.raises(ValueError, match="No valid values found in mask"):
        rioda.raster.clip_mask(xr.zeros_like(rioda))
    with pytest.raises(ValueError, match="should be geopandas"):
        rioda.raster.clip_geom(rioda.raster.bounds)


def test_reproject():
    # create data
    kwargs = dict(name="test", crs=4326)
    transform, shape = testdata[1][0], (9, 5, 5)
    da0 = raster.full_from_transform(transform, shape, **kwargs)
    da0.data = np.random.random(da0.shape)
    ds0 = da0.to_dataset()
    ds1 = raster.full_from_transform(*testdata[1], **kwargs).to_dataset()
    da2 = raster.full_from_transform(*testdata[3], **kwargs).to_dataset()
    assert np.all(ds1.raster.bounds == ds1.raster.transform_bounds(ds1.raster.crs))
    # test out of bounds -> return empty grid
    ds2_empty = ds0.raster.reproject_like(da2)
    assert ds2_empty.raster.identical_grid(da2)
    assert np.all(np.isnan(ds2_empty))
    assert ds2_empty.data_vars.keys() == ds0.data_vars.keys()
    da2_empty = da0.raster.reproject_like(da2)
    assert np.all(np.isnan(da2_empty))
    assert da2_empty.raster.identical_grid(da2)
    assert da2_empty.name == da0.name
    assert da2_empty.dtype == da0.dtype
    # flipud
    assert ds1.raster.flipud().raster.res[1] == -ds1.raster.res[1]
    # reproject nearest index
    ds2 = ds1.raster.reproject(dst_crs=3857, method="nearest_index")
    assert ds2.raster.crs.to_epsg() == 3857
    ds2 = ds1.raster.reproject(dst_crs=3857, dst_res=1000, align=True)
    assert np.all(np.asarray(ds2.raster.bounds)) // 1000 == 0
    ds2 = ds1.raster.reproject(dst_width=4, dst_height=2, method="average")
    assert np.all(ds2.raster.shape == (2, 4))
    ds2 = ds1.raster.reproject_like(ds0)
    assert np.all(ds2.raster.xcoords == ds0.raster.xcoords)
    ds2 = da0.raster.reproject_like(ds1, method="nearest_index")
    assert np.all(ds2.raster.xcoords == ds0.raster.xcoords)
    ds2 = ds1.raster.reproject(dst_crs="utm")
    assert ds2.raster.crs.is_projected
    index = ds1.raster.nearest_index(dst_crs="utm")
    assert isinstance(index, xr.DataArray)
    assert np.all(index.values >= -1)  ## -1 for invalid indices (outside domain)
    ds2_index = ds1.raster.reindex2d(index)
    assert np.all([np.all(ds2_index[c] == ds2[c]) for c in ds2_index.coords])
    # test with chunks
    da2_lazy = da0.chunk({da0.raster.dim0: 3}).raster.reproject(dst_crs="utm")
    assert isinstance(da2_lazy.data, dask.array.core.Array)
    assert np.all(ds2["test"] == da2_lazy.compute())
    da2_lazy = da0.chunk({da0.raster.dim0: 3}).raster.reproject(
        dst_crs="utm", method="nearest_index"
    )
    assert isinstance(da2_lazy.data, dask.array.core.Array)
    assert np.all(ds2_index["test"] == da2_lazy.compute())
    # make sure spatial ref is not lazy
    assert not isinstance(da2_lazy.spatial_ref.data, dask.array.Array)
    # check error messages
    with pytest.raises(ValueError, match="Resampling method unknown"):
        ds1.raster.reproject(dst_crs=3857, method="unknown")
    with pytest.raises(ValueError, match="CRS is missing"):
        ds1.drop_vars("spatial_ref").raster.reproject(dst_crs=3857)


def test_area_grid(rioda):
    # latlon
    area = rioda.raster.area_grid()
    assert area.std() > 0  # cells have different area
    # test dataset
    assert np.all(rioda.to_dataset().raster.area_grid() == area)
    # test projected crs
    rioda_proj = rioda.copy()
    rioda_proj.raster.set_crs(3857)
    area1 = rioda_proj.raster.area_grid()
    assert np.all(area1 == 0.25)
    # density
    assert np.all(rioda.raster.density_grid() == rioda / area)
    assert np.all(rioda_proj.raster.density_grid() == rioda_proj / area1)


def test_interpolate_na():
    # nodata is nan
    da0 = raster.full_from_transform(*testdata[0], nodata=np.nan)
    da0.values.flat[np.array([0, 3, -6])] = np.array([1, 1, 2])
    # default nearest interpolation
    da1 = da0.raster.interpolate_na()
    assert np.all(~np.isnan(da1))
    assert np.all(np.isin(da1, [1, 2]))
    # extra keyword argument to rasterio.fill.fillnodata
    da1 = da0.raster.interpolate_na(method="rio_idw", max_search_distance=5)
    assert np.all(~np.isnan(da1))
    # linear interpolation -> still nans
    da1 = da0.raster.interpolate_na(method="linear", extrapolate=False)
    assert np.isnan(da1).sum() == 14
    # extrapolate
    da1 = da0.raster.interpolate_na(method="linear", extrapolate=True)
    assert np.all(~np.isnan(da1))
    # with extra dimension
    da2 = da0.copy()  # adding extra dims to spatial_ref is done inplace
    da1 = da2.expand_dims("t").raster.interpolate_na()
    assert np.all(~np.isnan(da1))
    # test with other nodata value
    da3 = da0.fillna(-9999).astype(np.int32)
    da3.raster.set_nodata(-9999)
    assert da3.raster.interpolate_na().dtype == np.int32
    with pytest.raises(ValueError, match="Nodata value nan of type float"):
        da3.raster.set_nodata(np.nan)


def test_vector_grid(rioda):
    # polygon
    gdf = rioda.raster.vector_grid()
    assert rioda.raster.size == gdf.index.size
    assert np.all(gdf.geometry.type == "Polygon")
    assert np.all(gdf.total_bounds == rioda.raster.bounds)
    # line
    gdf = rioda.raster.vector_grid(geom_type="line")
    nrow, ncol = rioda.raster.shape
    assert gdf.index.size == (nrow + 1 + ncol + 1)
    assert np.all(gdf.geometry.type == "LineString")
    assert np.all(gdf.total_bounds == rioda.raster.bounds)
    # point
    gdf = rioda.raster.vector_grid(geom_type="point")
    assert np.all(gdf.geometry.type == "Point")
    assert rioda.raster.size == gdf.index.size
    assert np.all(gdf.intersects(rioda.raster.box.geometry[0]))


def test_sample():
    transform, shape = [0.2, 0.0, 3.0, 0.0, 0.25, -11.0], (8, 10)
    da = raster.full_from_transform(transform, shape, name="test", crs=4326)
    da.data = np.arange(da.raster.size).reshape(da.raster.shape).astype(da.dtype)
    gdf0 = gpd.GeoDataFrame(geometry=gpd.points_from_xy(da.x.values[:8], da.y.values))

    values = np.arange(0, 78, 11)
    assert np.all(da.to_dataset().raster.sample(gdf0)["test"] == values)

    nneighbors = np.array([4, 9, 9, 9, 9, 9, 9, 6])
    assert np.all(da.raster.sample(gdf0, wdw=1).count("wdw") == nneighbors)
    assert np.all(
        da.raster.sample(gdf0, wdw=1).median("wdw").values[1:-1] == values[1:-1]
    )

    with pytest.raises(ValueError, match="Only point geometries accepted"):
        da.raster.sample(gdf0.buffer(1))


def test_zonal_stats():
    transform, shape = [0.2, 0.0, 3.0, 0.0, -0.25, -11.0], (8, 10)
    da = raster.full_from_transform(transform, shape, name="test", crs=4326)
    da.data = np.arange(da.raster.size).reshape(da.raster.shape).astype(da.dtype)
    ds = xr.merge([da, da.expand_dims("time").to_dataset().rename({"test": "test1"})])
    w, s, e, n = da.raster.bounds
    geoms = [
        box(w, s, w + abs(e - w) / 2.0, n),
        box(w - 2, s, w - 0.2, n),  # outside
        Point((w + 0.1, n - 0.1)),
        LineString([(w, (n + s) / 2 - 0.1), (e, (n + s) / 2 - 0.1)]),  # vert line
    ]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=da.raster.crs)

    ds0 = da.raster.zonal_stats(gdf, "count")
    assert "test_count" in ds0.data_vars
    assert "index" in ds0.dims
    assert np.all(ds0["index"] == np.array([0, 2, 3]))
    assert np.all(ds0["test_count"] == np.array([40, 1, 10]))

    ds0 = ds.raster.zonal_stats(gdf.to_crs(3857), [np.nanmean, "mean"])
    ds1 = ds.raster.zonal_stats(gdf, [np.nanmean, "mean"])

    assert np.all(ds0["test_nanmean"] == ds0["test_mean"])
    assert np.all(ds1["test_mean"] == ds0["test_mean"])

    with pytest.raises(ValueError, match="Stat asdf not valid"):
        da.raster.zonal_stats(gdf, "asdf")
    with pytest.raises(IndexError, match="All geometries outside raster domain"):
        da.raster.zonal_stats(gdf.iloc[1:2], "mean")


@pytest.mark.parametrize(("transform", "shape"), testdata[-2:])
def test_rotated(transform, shape, tmpdir):
    da = raster.full_from_transform(transform, shape, nodata=-1, name="test")
    da.raster.set_crs(4326)
    da[:] = 1
    # test I/O
    path = str(tmpdir.join("rotated.tif"))
    da.raster.to_raster(path)
    assert da.raster.identical_grid(open_raster(path))
    # test rasterize
    gdf = da.raster.vector_grid()
    gdf["value"] = np.arange(gdf.index.size).astype(np.float32)
    da2 = da.raster.rasterize(gdf, col_name="value", nodata=-1)
    assert np.all(da2.values.flatten() == gdf["value"])
    # test vectorize
    gdf2 = da2.raster.vectorize().sort_values("value")
    gdf2.index = gdf2.index.astype(int)
    gpd.testing.assert_geodataframe_equal(
        gdf, gdf2, check_less_precise=True, check_dtype=False, check_index_type=False
    )
    # test sample
    idxs = np.array([2, 7])
    pnts = gpd.points_from_xy(*da.raster.idx_to_xy(idxs))
    gdf_pnts = gpd.GeoDataFrame(geometry=pnts, crs=4326)
    assert np.all(idxs == da2.raster.sample(gdf_pnts))
    # zonal stat
    assert np.all(idxs == da2.raster.zonal_stats(gdf_pnts, ["mean"])["value_mean"])
    # test reproject to non-rotated utm grid
    dst_crs = gis_utils.parse_crs("utm", da.raster.bounds)
    da2_reproj = da2.raster.reproject(dst_crs=dst_crs)
    assert np.all(da2.raster.box.intersects(da2_reproj.raster.box.to_crs(4326)))


def test_to_xyz_tiles(tmpdir, rioda_large):
    path = str(tmpdir)
    rioda_large.raster.to_xyz_tiles(
        join(path, "dummy_xyz"),
        tile_size=256,
        zoom_levels=[0, 2],
    )
    with rasterio.open(join(path, "dummy_xyz", "dummy_xyz_zl0.vrt"), "r") as src:
        assert src.shape == (1024, 1024)
    with rasterio.open(join(path, "dummy_xyz", "dummy_xyz_zl2.vrt"), "r") as src:
        assert src.shape == (256, 256)

    test_bounds = [2.13, -2.13, 3.2, -1.07]
    _test_r = open_raster(join(path, "dummy_xyz", "0", "2", "1.tif"))
    assert [round(_n, 2) for _n in _test_r.raster.bounds] == test_bounds


def test_to_raster(rioda: xr.DataArray, tmp_dir: Path):
    uri_tif = str(tmp_dir / "test_to_raster.tif")
    rioda.raster.to_raster(uri_tif)
    assert Path(uri_tif).is_file()


def test_to_raster_raises_on_invalid_kwargs(rioda: xr.DataArray, tmp_dir: Path):
    with pytest.raises(ValueError, match="will be set based on the DataArray"):
        rioda.raster.to_raster(str(tmp_dir / "test2.tif"), count=3)


def test_to_mapstack_raises_on_invalid_driver(rioda: xr.DataArray, tmp_dir: Path):
    with pytest.raises(ValueError, match="Extension unknown for driver"):
        rioda.to_dataset().raster.to_mapstack(root=str(tmp_dir), driver="unknown")


def test_to_mapstack(rioda: xr.DataArray, tmp_dir: Path):
    ds = rioda.to_dataset()
    prefix = "_test_"
    ds.raster.to_mapstack(str(tmp_dir), prefix=prefix, mask=True, driver="GTiff")
    for name in ds.raster.vars:
        assert (tmp_dir / f"{prefix}{name}.tif").is_file()


def test_to_slippy_tiles(tmpdir, rioda_large):
    from PIL import Image

    # for tile at zl 7, x 64, y 64
    test_bounds = [0.0, -313086.07, 313086.07, -0.0]
    # populate with random data
    np.random.seed(0)
    rioda_large[:] = np.random.random(rioda_large.shape).astype(np.float32)

    # png
    png_dir = join(tmpdir, "tiles_png")
    rioda_large.raster.to_slippy_tiles(png_dir)
    _zl = os.listdir(png_dir)
    _zl = [int(_n) for _n in _zl]
    assert len(_zl) == 4
    assert min(_zl) == 6
    assert max(_zl) == 9
    path = join(png_dir, "7", "64", "64.png")
    im = np.array(Image.open(path))
    assert im.shape == (256, 256, 4)
    assert all(im[0, 0, :] == [128, 0, 132, 255])

    # test with cmap
    png_dir = join(tmpdir, "tiles_png_cmap")
    rioda_large.raster.to_slippy_tiles(png_dir, cmap="viridis", min_lvl=6, max_lvl=7)
    path = join(png_dir, "7", "64", "64.png")
    im = np.array(Image.open(path))
    assert im.shape == (256, 256, 4)
    assert all(im[0, 0, :] == [32, 143, 140, 255])

    # gtiff
    tif_dir = join(tmpdir, "tiles_tif")
    rioda_large.raster.to_slippy_tiles(
        tif_dir,
        driver="GTiff",
        min_lvl=5,
        max_lvl=8,
        write_vrt=True,
    )
    with rasterio.open(join(tif_dir, "lvl5.vrt"), "r") as src:
        assert src.shape == (256, 256)
    with rasterio.open(join(tif_dir, "lvl8.vrt"), "r") as src:
        assert src.shape == (1024, 768)
    _test_r = open_raster(join(tif_dir, "7", "64", "64.tif"))
    assert [round(_n, 2) for _n in _test_r.raster.bounds] == test_bounds
    assert all([isfile(join(tif_dir, f"lvl{zl}.vrt")) for zl in range(5, 9)])
    _test_vrt = open_raster(join(tif_dir, "lvl7.vrt"))
    assert isinstance(_test_vrt, xr.DataArray)

    # nc
    nc_dir = join(tmpdir, "tiles_nc")
    rioda_large.raster.to_slippy_tiles(
        nc_dir,
        driver="netcdf4",
        min_lvl=5,
        max_lvl=8,
        write_vrt=True,
    )
    with rasterio.open(join(nc_dir, "lvl5.vrt"), "r") as src:
        assert src.shape == (256, 256)
    with rasterio.open(join(nc_dir, "lvl8.vrt"), "r") as src:
        assert src.shape == (1024, 768)
    _test_r = open_raster(join(nc_dir, "7", "64", "64.nc"))
    assert [round(_n, 2) for _n in _test_r.raster.bounds] == test_bounds
    assert all([isfile(join(nc_dir, f"lvl{zl}.vrt")) for zl in range(5, 9)])
    _test_vrt = open_raster(join(nc_dir, "lvl7.vrt"))
    assert isinstance(_test_vrt, xr.DataArray)

    # test all errors in to_slippy_tiles
    with pytest.raises(ValueError, match="Unkown file driver"):
        rioda_large.raster.to_slippy_tiles(str(tmpdir), driver="unsupported")
    with pytest.raises(ValueError, match="Only 2d DataArrays"):
        rioda_large.expand_dims("t").raster.to_slippy_tiles(str(tmpdir))
    with pytest.raises(ValueError, match="Colormap is only supported for png"):
        rioda_large.raster.to_slippy_tiles(str(tmpdir), cmap="viridis", driver="GTiff")
