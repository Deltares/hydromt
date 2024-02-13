# -*- coding: utf-8 -*-
"""Tests for the io submodule."""

import glob
import os
from os.path import join
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from shapely.geometry import box

import hydromt
from hydromt import _compat, raster


@pytest.mark.parametrize("engine", ["fiona", "pyogrio"])
def test_open_vector(engine, tmpdir, df, geodf, world):
    gpd.io_engine = engine
    fn_csv = str(tmpdir.join("test.csv"))
    fn_parquet = str(tmpdir.join("test.parquet"))
    fn_xy = str(tmpdir.join("test.xy"))
    fn_xls = str(tmpdir.join("test.xlsx"))
    fn_geojson = str(tmpdir.join("test.geojson"))
    fn_shp = str(tmpdir.join("test.shp"))
    fn_gpkg = str(tmpdir.join("test.gpkg"))
    df.to_csv(fn_csv)
    df.to_parquet(fn_parquet)
    if _compat.HAS_OPENPYXL:
        df.to_excel(fn_xls)
    geodf.to_file(fn_geojson, driver="GeoJSON")
    hydromt.write_xy(fn_xy, geodf)
    geodf.to_file(fn_shp)
    geodf.to_file(fn_gpkg, driver="GPKG")
    # read csv
    gdf1 = hydromt.open_vector(fn_csv, assert_gtype="Point", crs=4326)
    assert gdf1.crs == geodf.crs
    assert np.all(gdf1 == geodf)
    # no data in domain
    gdf1 = hydromt.open_vector(fn_csv, crs=4326, bbox=[200, 300, 200, 300])
    assert gdf1.index.size == 0
    if _compat.HAS_OPENPYXL:
        # read xls
        gdf1 = hydromt.open_vector(fn_xls, assert_gtype="Point", crs=4326)
        assert np.all(gdf1 == geodf)

    # read xy
    gdf1 = hydromt.open_vector(fn_xy, crs=4326)
    assert np.all(gdf1 == geodf[["geometry"]])
    # read shapefile
    gdf1 = hydromt.open_vector(fn_shp, bbox=list(geodf.total_bounds))
    assert np.all(gdf1 == geodf)
    mask = gpd.GeoDataFrame({"geometry": [box(*geodf.total_bounds)]}, crs=geodf.crs)
    gdf1 = hydromt.open_vector(fn_shp, geom=mask)
    assert np.all(gdf1 == geodf)
    # read geopackage
    gdf1 = hydromt.open_vector(fn_gpkg)
    assert np.all(gdf1 == geodf)
    # read parquet
    gdf1 = hydromt.open_vector(fn_parquet, crs=4326)
    assert np.all(gdf1 == geodf)
    # filter
    country = "Chile"
    geom = world[world["name"] == country]
    gdf1 = hydromt.open_vector(
        fn_csv, crs=4326, geom=geom.to_crs(3857)
    )  # crs should default to 4326
    assert np.all(gdf1["country"] == country)
    gdf2 = hydromt.open_vector(fn_geojson, geom=geom)
    # NOTE labels are different
    assert np.all(gdf1.geometry.values == gdf2.geometry.values)
    gdf2 = hydromt.open_vector(fn_csv, crs=4326, bbox=geom.total_bounds)
    assert np.all(gdf1.geometry.values == gdf2.geometry.values)
    # error
    with pytest.raises(ValueError, match="other geometries"):
        hydromt.open_vector(fn_csv, assert_gtype="Polygon")
    with pytest.raises(ValueError, match="unknown"):
        hydromt.open_vector(fn_csv, assert_gtype="PolygonPoints")
    with pytest.raises(ValueError, match="The GeoDataFrame has no CRS"):
        hydromt.open_vector(fn_csv)
    with pytest.raises(ValueError, match="Unknown geometry mask type"):
        hydromt.open_vector(fn_csv, crs=4326, geom=geom.total_bounds)
    with pytest.raises(ValueError, match="x dimension"):
        hydromt.open_vector(fn_csv, x_dim="x")
    with pytest.raises(ValueError, match="y dimension"):
        hydromt.open_vector(fn_csv, y_dim="y")
    with pytest.raises(IOError, match="No such file"):
        hydromt.open_vector("fail.csv")
    with pytest.raises(IOError, match="Driver fail unknown"):
        hydromt.open_vector_from_table("test.fail")


def test_open_geodataset(tmpdir, geodf):
    fn_gdf = str(tmpdir.join("points.geojson"))
    geodf.to_file(fn_gdf, driver="GeoJSON")
    # create equivalent polygon file
    fn_gdf_poly = str(tmpdir.join("polygons.geojson"))
    geodf_poly = geodf.copy()
    geodf_poly["geometry"] = geodf_poly.buffer(0.1)
    geodf_poly.to_file(fn_gdf_poly, driver="GeoJSON")
    # create zeros timeseries
    ts = pd.DataFrame(
        index=pd.DatetimeIndex(["01-01-2000", "01-01-2001"]),
        columns=geodf.index.values,
        data=np.zeros((2, geodf.index.size)),
    )
    name = "waterlevel"
    fn_ts = str(tmpdir.join(f"{name}.csv"))
    ts.to_csv(fn_ts)
    # returns dataset with coordinates, but no variable
    ds = hydromt.open_geodataset(fn_gdf)
    assert isinstance(ds, xr.Dataset)
    assert len(ds.data_vars) == 0
    geodf1 = ds.vector.to_gdf()
    assert np.all(geodf == geodf1[geodf.columns])
    # add timeseries
    ds = hydromt.open_geodataset(fn_gdf, fn_ts)
    assert name in ds.data_vars
    assert np.all(ds[name].values == 0)
    # test for polygon geometry
    ds = hydromt.open_geodataset(fn_gdf_poly, fn_ts)
    assert name in ds.data_vars
    assert ds.vector.geom_type == "Polygon"
    with pytest.raises(IOError, match="GeoDataset point location file not found"):
        hydromt.open_geodataset("missing_file.csv")
    with pytest.raises(IOError, match="GeoDataset data file not found"):
        hydromt.open_geodataset(fn_gdf, fn_data="missing_file.csv")


def test_timeseries_io(tmpdir, ts):
    fn_ts = str(tmpdir.join("test1.csv"))
    # dattime in columns
    ts.to_csv(fn_ts)
    da = hydromt.open_timeseries_from_table(fn_ts)
    assert isinstance(da, xr.DataArray)
    assert da.time.dtype.type.__name__ == "datetime64"
    # transposed df > datetime in row index
    fn_ts2 = str(tmpdir.join("test2.csv"))
    ts = ts.T
    ts.to_csv(fn_ts2)
    da2 = hydromt.open_timeseries_from_table(fn_ts2)
    assert da.time.dtype.type.__name__ == "datetime64"
    assert np.all(da == da2)
    # no time index
    fn_ts3 = str(tmpdir.join("test3.csv"))
    pd.DataFrame(ts.values).to_csv(fn_ts3)
    with pytest.raises(ValueError, match="No time index found"):
        hydromt.open_timeseries_from_table(fn_ts3)
    # parse str index to numeric index
    cols = [f"a_{i}" for i in ts.columns]
    ts.columns = cols
    fn_ts4 = str(tmpdir.join("test4.csv"))
    ts.to_csv(fn_ts4)
    da4 = hydromt.open_timeseries_from_table(fn_ts4)
    assert np.all(da == da4)
    assert np.all(da.index == da4.index)
    # no numeric index
    cols[0] = "a"
    ts.columns = cols
    fn_ts5 = str(tmpdir.join("test5.csv"))
    ts.to_csv(fn_ts5)
    with pytest.raises(ValueError, match="No numeric index"):
        hydromt.open_timeseries_from_table(fn_ts5)


def test_raster_io(tmpdir, rioda):
    da = rioda
    fn_tif = str(tmpdir.join("test.tif"))
    # to_raster / open_raster
    da.raster.to_raster(fn_tif, crs=3857, tags={"name": "test"})
    assert os.path.isfile(fn_tif)
    assert np.all(hydromt.open_raster(fn_tif).values == da.values)
    with rasterio.open(fn_tif, "r") as src:
        assert src.tags()["name"] == "test"
        assert src.crs.to_epsg() == 3857
    da1 = hydromt.open_raster(fn_tif, mask_nodata=True)
    assert np.any(np.isnan(da1.values))
    # TODO window needs checking & better testing
    fn_tif = str(tmpdir.join("test1.tif"))
    da1.raster.to_raster(fn_tif, nodata=-9999, windowed=True)
    da2 = hydromt.open_raster(fn_tif)
    assert not np.any(np.isnan(da2.values))
    fn_tif = str(tmpdir.join("test_2.tif"))
    da1.fillna(da1.attrs["_FillValue"]).expand_dims("t").round(0).astype(
        np.int32
    ).raster.to_raster(fn_tif, dtype=np.int32)
    da3 = hydromt.open_raster(fn_tif)
    assert da3.dtype == np.int32
    # to_mapstack / open_mfraster
    ds = da.to_dataset()
    prefix = "_test_"
    root = str(tmpdir)
    ds.raster.to_mapstack(root, prefix=prefix, mask=True, driver="GTiff")
    for name in ds.raster.vars:
        assert os.path.isfile(join(root, f"{prefix}{name}.tif"))
    ds_in = hydromt.open_mfraster(join(root, f"{prefix}*.tif"), mask_nodata=True)
    dvars = ds_in.raster.vars
    assert np.all([n in dvars for n in ds.raster.vars])
    assert np.all([np.isnan(ds_in[n].raster.nodata) for n in dvars])
    # concat
    fn_tif = str(tmpdir.join("test_3.tif"))
    da.raster.to_raster(fn_tif, crs=3857)
    ds_in = hydromt.open_mfraster(join(root, "test_*.tif"), concat=True)
    assert ds_in[ds_in.raster.vars[0]].ndim == 3
    # with reading with pathlib
    paths = [Path(p) for p in glob.glob(join(root, f"{prefix}*.tif"))]
    dvars2 = hydromt.open_mfraster(paths, mask_nodata=True).raster.vars
    assert np.all([f"{prefix}{n}" in dvars2 for n in ds.raster.vars])
    # test writing to subdir
    ds.rename({"test": "test/test"}).raster.to_mapstack(root, driver="GTiff")
    assert os.path.isfile(join(root, "test", "test.tif"))


def test_open_mfcsv_by_id(tmpdir, dfs_segmented_by_points):
    df_fns = {
        i: str(tmpdir.join("data", f"{i}.csv"))
        for i in range(len(dfs_segmented_by_points))
    }
    os.mkdir(tmpdir.join("data"))
    for i in range(len(df_fns)):
        dfs_segmented_by_points[i].to_csv(df_fns[i])

    ds = hydromt.io.open_mfcsv(df_fns, "id")

    assert sorted(list(ds.data_vars.keys())) == ["test1", "test2"], ds
    assert sorted(list(ds.dims)) == ["id", "time"], ds
    for i in range(len(dfs_segmented_by_points)):
        test1 = ds.sel(id=i)["test1"]
        test2 = ds.sel(id=i)["test2"]
        assert np.all(
            np.equal(test1, np.arange(len(dfs_segmented_by_points)) * i)
        ), test1
        assert np.all(
            np.equal(test2, np.arange(len(dfs_segmented_by_points)) ** i)
        ), test2

    # again but with a nameless csv index
    for i in range(len(df_fns)):
        dfs_segmented_by_points[i].rename_axis(None, axis=0, inplace=True)
        dfs_segmented_by_points[i].to_csv(df_fns[i])

    ds = hydromt.io.open_mfcsv(df_fns, "id")

    assert sorted(list(ds.data_vars.keys())) == ["test1", "test2"], ds
    assert sorted(list(ds.dims)) == ["id", "index"], ds
    for i in range(len(dfs_segmented_by_points)):
        test1 = ds.sel(id=i)["test1"]
        test2 = ds.sel(id=i)["test2"]
        assert np.all(
            np.equal(test1, np.arange(len(dfs_segmented_by_points)) * i)
        ), test1
        assert np.all(
            np.equal(test2, np.arange(len(dfs_segmented_by_points)) ** i)
        ), test2


def test_open_mfcsv_by_var(tmpdir, dfs_segmented_by_vars):
    os.mkdir(tmpdir.join("data"))
    fns = {}
    for var, df in dfs_segmented_by_vars.items():
        fn = tmpdir.join("data", f"{var}.csv")
        df.to_csv(fn)
        fns[var] = fn

    ds = hydromt.io.open_mfcsv(fns, "id", segmented_by="var")

    assert sorted(list(ds.data_vars.keys())) == ["test1", "test2"], ds
    ids = ds.id.values
    for i in ids:
        test1 = ds.sel(id=i)["test1"]
        test2 = ds.sel(id=i)["test2"]
        assert np.all(np.equal(test1, np.arange(len(ids)) * int(i))), test1
        assert np.all(np.equal(test2, np.arange(len(ids)) ** int(i))), test2


def test_rasterio_errors(tmpdir, rioda):
    with pytest.raises(OSError, match="no files to open"):
        hydromt.open_mfraster(str(tmpdir.join("test*.tiffff")))
    da0 = raster.full_from_transform(
        [0.5, 0.0, 3.0, 0.0, -0.5, -9.0], (4, 6), nodata=-1, name="test"
    )
    da1 = raster.full_from_transform(
        [0.2, 0.0, 3.0, 0.0, 0.25, -11.0], (8, 15), nodata=-1, name="test"
    )
    da0.raster.to_raster(str(tmpdir.join("test0.tif")))
    da1.raster.to_raster(str(tmpdir.join("test1.tif")))
    with pytest.raises(xr.MergeError, match="Geotransform and/or shape do not match"):
        hydromt.open_mfraster(str(tmpdir.join("test*.tif")))
    with pytest.raises(ValueError, match="will be set based on the DataArray"):
        da0.raster.to_raster(str(tmpdir.join("test2.tif")), count=3)
    with pytest.raises(ValueError, match="Extension unknown for driver"):
        da0.to_dataset().raster.to_mapstack(root=str(tmpdir), driver="unknown")
