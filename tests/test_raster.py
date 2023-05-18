# -*- coding: utf-8 -*-
"""Tests for the hydromt.raster submodule."""

import os

import dask
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from osgeo import gdal
from shapely.geometry import LineString, Point, box

from hydromt import gis_utils, open_raster, raster

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


@pytest.mark.parametrize("origin, rotation, res, shape, bounds", tests)
def test_raster_properties(origin, rotation, res, shape, bounds):
    transform = get_transform(origin, rotation, res)
    da = raster.full_from_transform(transform, shape, name="test", crs=4326)
    assert np.allclose(rotation, da.raster.rotation)
    assert np.allclose(res, da.raster.res)
    assert np.allclose(origin, da.raster.origin)
    assert np.allclose(transform, da.raster.transform)
    assert np.allclose(da.raster.box.total_bounds, da.raster.bounds)
    assert np.allclose(bounds, da.raster.internal_bounds)


@pytest.mark.parametrize("transform, shape", testdata)
def test_attrs(transform, shape):
    # checks on raster spatial attributes
    da = raster.full_from_transform(transform, shape, name="test")
    da.drop_vars(raster.GEO_MAP_COORD)  # reset attrs
    assert isinstance(da.raster.attrs, dict)
    assert raster.GEO_MAP_COORD in da.coords
    assert da.raster.dims == ("y", "x")
    assert "x_dim" in da.raster.attrs
    assert da.raster.dim0 == None if len(shape) == 2 else "dim0"
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


def test_gdal(tmpdir):
    da = raster.full_from_transform(*testdata[0], name="test")
    # Add crs
    da.attrs.update(crs=4326)
    da.raster.set_crs()
    # Update gdal compliant attrs
    da1 = da.raster.gdal_compliant(rename_dims=True, force_sn=True)
    assert raster.GEO_MAP_COORD in da1.coords
    assert da1.raster.dims == ("lat", "lon")
    assert da1.raster.res[1] > 0
    # Update without rename and SN orientation
    da = da.raster.gdal_compliant(rename_dims=False, force_sn=False)
    assert da.raster.dims == ("y", "x")
    assert da.raster.res[1] < 0
    # Write to netcdf and reopen with gdal
    fn_nc = str(tmpdir.join("gdal_test.nc"))
    da.to_netcdf(fn_nc)
    info = gdal.Info(fn_nc)
    ds = gdal.Open(fn_nc)
    assert da[raster.GEO_MAP_COORD].attrs["crs_wkt"] == ds.GetProjection()


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
    rioda["x"] = np.random.rand(rioda["x"].size)
    with pytest.raises(ValueError, match="only applies to regular grids"):
        rioda.raster.set_spatial_dims()
    with pytest.raises(ValueError, match="Invalid dimension order."):
        rioda.transpose("x", "y").raster._check_dimensions()
    with pytest.raises(ValueError, match="Invalid dimension order."):
        da1 = rioda.expand_dims("t").transpose("y", "x", "t", transpose_coords=True)
        da1.raster._check_dimensions()
    with pytest.raises(ValueError, match="Only 2D and 3D data"):
        rioda.expand_dims(("t", "t1")).raster._check_dimensions()


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


@pytest.mark.parametrize("transform, shape", testdata)
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

    da = rioda.raster.rasterize_geometry(gdf, method="fraction", name="frac", nodata=0)
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


@pytest.mark.parametrize("transform, shape", testdata)
def test_clip(transform, shape):
    # create rasterdataarray with crs
    da = raster.full_from_transform(transform, shape, nodata=1, name="test", crs=4326)
    da.raster.set_nodata(0)
    # create gdf covering approx half raster
    w, s, _, n = da.raster.bounds
    e, _ = da.raster.transform * (shape[1] // 2, shape[0] // 2)
    gdf = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs=da.raster.crs)
    # test bbox - buffer
    da_clip = da.raster.clip_bbox(gdf.total_bounds, buffer=da.raster.width)
    assert np.all(np.isclose(da.raster.bounds, da_clip.raster.bounds))
    # test bbox
    da_clip0 = da.raster.clip_bbox(gdf.total_bounds)
    # test geom
    da_clip1 = da.raster.clip_geom(gdf)
    assert np.all(np.isclose(da_clip1.raster.bounds, da_clip0.raster.bounds))
    # test mask
    da_clip1 = da.raster.clip_mask(da.raster.geometry_mask(gdf))
    assert np.all(np.isclose(da_clip1.raster.bounds, da_clip0.raster.bounds))
    # test geom - different crs
    da_clip1 = da.raster.clip_geom(gdf.to_crs(3857))
    assert np.all(np.isclose(da_clip1.raster.bounds, da_clip0.raster.bounds))

    # these test are for non-rotated only
    if da.raster.rotation != 0:
        return
    assert np.all(np.isclose(da_clip0.raster.bounds, gdf.total_bounds))
    # test bbox - align
    align = np.round(abs(da.raster.res[0] * 2), 2)
    da_clip = da.raster.clip_bbox(gdf.total_bounds, align=align)
    dalign = np.round(da_clip.raster.bounds[2], 2) % align
    assert np.isclose(dalign, 0) or np.isclose(dalign, align)


def test_clip_errors(rioda):
    with pytest.raises(ValueError, match="Mask should be xarray.DataArray type."):
        rioda.raster.clip_mask(rioda.values)
    with pytest.raises(ValueError, match="Mask shape invalid."):
        rioda.raster.clip_mask(rioda.isel({"x": slice(1, -1)}))
    with pytest.raises(ValueError, match="Invalid mask."):
        rioda.raster.clip_mask(xr.zeros_like(rioda))
    with pytest.raises(ValueError, match="should be geopandas"):
        rioda.raster.clip_geom(rioda.raster.bounds)


def test_reproject():
    kwargs = dict(name="test", crs=4326)
    transform, shape = testdata[1][0], (9, 5, 5)
    da0 = raster.full_from_transform(transform, shape, **kwargs)
    da0.data = np.random.random(da0.shape)
    ds0 = da0.to_dataset()
    ds1 = raster.full_from_transform(*testdata[1], **kwargs).to_dataset()
    assert np.all(ds1.raster.bounds == ds1.raster.transform_bounds(ds1.raster.crs))
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


def test_vector_grid(rioda):
    gdf = rioda.raster.vector_grid()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert rioda.raster.size == gdf.index.size


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
    assert "test_count" in ds0.data_vars and "index" in ds0.dims
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


@pytest.mark.parametrize("transform, shape", testdata[-2:])
def test_rotated(transform, shape, tmpdir):
    da = raster.full_from_transform(transform, shape, nodata=-1, name="test")
    da.raster.set_crs(4326)
    da[:] = 1
    # test I/O
    fn = str(tmpdir.join("rotated.tif"))
    da.raster.to_raster(fn)
    assert da.raster.identical_grid(open_raster(fn))
    # test rasterize
    gdf = da.raster.vector_grid()
    gdf["value"] = np.arange(gdf.index.size).astype(np.float32)
    da2 = da.raster.rasterize(gdf, col_name="value", nodata=-1)
    assert np.all(da2.values.flatten() == gdf["value"])
    # test vectorize
    gdf2 = da2.raster.vectorize().sort_values("value")
    gdf2.index = gdf2.index.astype(int)
    gpd.testing.assert_geodataframe_equal(
        gdf, gdf2, check_less_precise=True, check_dtype=False
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
    # NOTE: this method does not work in debug mode because of os.subprocess
    path = str(tmpdir)
    rioda_large.raster.to_xyz_tiles(
        os.path.join(path, "dummy_xyz"),
        tile_size=256,
        zoom_levels=[0, 2],
    )
    with open(os.path.join(path, "dummy_xyz", "0", "filelist.txt"), "r") as f:
        assert len(f.readlines()) == 16
    with open(os.path.join(path, "dummy_xyz", "2", "filelist.txt"), "r") as f:
        assert len(f.readlines()) == 1


# def test_to_osm(tmpdir, dummy):
#     path = str(tmpdir)
#     dummy.raster.to_osm(
#         f"{path}\\dummy_osm",
#         zl=4,
#         bbox=(0, -45, 45, 0),
#     )
#     f = open(f"{path}\\dummy_osm\\3\\filelist.txt", "r")
#     assert len(f.readlines()) == 4
