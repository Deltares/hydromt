# -*- coding: utf-8 -*-
"""Tests for the io submodule."""

import os
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box
from upath import UPath

import hydromt
from hydromt import _compat
from hydromt.io.readers import (
    open_geodataset,
    open_mfcsv,
    open_timeseries_from_table,
    open_vector,
    open_vector_from_table,
)
from hydromt.io.writers import write_xy


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
    write_xy(fn_xy, geodf)
    geodf.to_file(fn_shp)
    geodf.to_file(fn_gpkg, driver="GPKG")
    # read csv
    gdf1 = open_vector(fn_csv, assert_gtype="Point", crs=4326)
    assert gdf1.crs == geodf.crs
    assert np.all(gdf1 == geodf)
    # no data in domain
    gdf1 = open_vector(fn_csv, crs=4326, bbox=[200, 300, 200, 300])
    assert gdf1.index.size == 0
    if _compat.HAS_OPENPYXL:
        # read xls
        gdf1 = open_vector(fn_xls, assert_gtype="Point", crs=4326)
        assert np.all(gdf1 == geodf)

    # read xy
    gdf1 = open_vector(fn_xy, crs=4326)
    assert np.all(gdf1 == geodf[["geometry"]])
    # read shapefile
    gdf1 = hydromt.io.open_vector(fn_shp, bbox=list(geodf.total_bounds))
    assert np.all(gdf1 == geodf)
    mask = gpd.GeoDataFrame({"geometry": [box(*geodf.total_bounds)]}, crs=geodf.crs)
    gdf1 = hydromt.io.open_vector(fn_shp, geom=mask)
    assert np.all(gdf1 == geodf)
    # read geopackage
    gdf1 = hydromt.io.open_vector(fn_gpkg)
    assert np.all(gdf1 == geodf)
    # read parquet
    gdf1 = open_vector(fn_parquet, crs=4326)
    assert np.all(gdf1 == geodf)
    # filter
    country = "Chile"
    geom = world[world["name"] == country]
    gdf1 = open_vector(
        fn_csv, crs=4326, geom=geom.to_crs(3857)
    )  # crs should default to 4326
    assert np.all(gdf1["country"] == country)
    gdf2 = open_vector(fn_geojson, geom=geom)
    # NOTE labels are different
    assert np.all(gdf1.geometry.values == gdf2.geometry.values)
    gdf2 = open_vector(fn_csv, crs=4326, bbox=geom.total_bounds)
    assert np.all(gdf1.geometry.values == gdf2.geometry.values)
    # error
    with pytest.raises(ValueError, match="other geometries"):
        open_vector(fn_csv, assert_gtype="Polygon")
    with pytest.raises(ValueError, match="unknown"):
        open_vector(fn_csv, assert_gtype="PolygonPoints")
    with pytest.raises(ValueError, match="The GeoDataFrame has no CRS"):
        open_vector(fn_csv)
    with pytest.raises(ValueError, match="Unknown geometry mask type"):
        open_vector(fn_csv, crs=4326, geom=geom.total_bounds)
    with pytest.raises(ValueError, match="x dimension"):
        open_vector(fn_csv, x_dim="x")
    with pytest.raises(ValueError, match="y dimension"):
        open_vector(fn_csv, y_dim="y")
    with pytest.raises(IOError, match="No such file"):
        open_vector("fail.csv")
    with pytest.raises(IOError, match="Driver fail unknown"):
        open_vector_from_table("test.fail")


@pytest.mark.skipif(not _compat.HAS_S3FS, reason="S3FS not installed.")
def test_open_vector_s3(geodf: gpd.GeoDataFrame):
    m = MagicMock()
    m.return_value = geodf
    with patch("geopandas.io.file._read_file_fiona", m):
        df = hydromt.io.open_vector(UPath("s3://fake_url/file.geojson"))
    assert np.all(geodf == df)


def test_open_geodataset(tmpdir, geodf):
    fn_gdf = str(tmpdir.join("points.geojson"))
    geodf.to_file(fn_gdf, driver="GeoJSON")
    # create equivalent polygon file
    fn_gdf_poly = str(tmpdir.join("polygons.geojson"))
    geodf_poly = geodf.copy()
    crs = geodf.crs
    geodf_poly["geometry"] = geodf_poly.to_crs(3857).buffer(0.1).to_crs(crs)
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
    ds = open_geodataset(fn_gdf)
    assert isinstance(ds, xr.Dataset)
    assert len(ds.data_vars) == 0
    geodf1 = ds.vector.to_gdf()
    assert np.all(geodf == geodf1[geodf.columns])
    # add timeseries
    ds = open_geodataset(fn_gdf, fn_ts)
    assert name in ds.data_vars
    assert np.all(ds[name].values == 0)
    # test for polygon geometry
    ds = open_geodataset(fn_gdf_poly, fn_ts)
    assert name in ds.data_vars
    assert ds.vector.geom_type == "Polygon"
    with pytest.raises(IOError, match="GeoDataset point location file not found"):
        open_geodataset("missing_file.csv")
    with pytest.raises(IOError, match="GeoDataset data file not found"):
        open_geodataset(fn_gdf, fn_data="missing_file.csv")


def test_timeseries_io(tmpdir, ts):
    fn_ts = str(tmpdir.join("test1.csv"))
    # dattime in columns
    ts.to_csv(fn_ts)
    da = open_timeseries_from_table(fn_ts)
    assert isinstance(da, xr.DataArray)
    assert da.time.dtype.type.__name__ == "datetime64"
    # transposed df > datetime in row index
    fn_ts2 = str(tmpdir.join("test2.csv"))
    ts = ts.T
    ts.to_csv(fn_ts2)
    da2 = open_timeseries_from_table(fn_ts2)
    assert da.time.dtype.type.__name__ == "datetime64"
    assert np.all(da == da2)
    # no time index
    fn_ts3 = str(tmpdir.join("test3.csv"))
    pd.DataFrame(ts.values).to_csv(fn_ts3)
    with pytest.raises(ValueError, match="No time index found"):
        open_timeseries_from_table(fn_ts3)
    # parse str index to numeric index
    cols = [f"a_{i}" for i in ts.columns]
    ts.columns = cols
    fn_ts4 = str(tmpdir.join("test4.csv"))
    ts.to_csv(fn_ts4)
    da4 = open_timeseries_from_table(fn_ts4)
    assert np.all(da == da4)
    assert np.all(da.index == da4.index)
    # no numeric index
    cols[0] = "a"
    ts.columns = cols
    fn_ts5 = str(tmpdir.join("test5.csv"))
    ts.to_csv(fn_ts5)
    with pytest.raises(ValueError, match="No numeric index"):
        open_timeseries_from_table(fn_ts5)


def test_open_mfcsv_by_id(tmpdir, dfs_segmented_by_points):
    df_fns = {
        i: str(tmpdir.join("data", f"{i}.csv"))
        for i in range(len(dfs_segmented_by_points))
    }
    os.mkdir(tmpdir.join("data"))
    for i in range(len(df_fns)):
        dfs_segmented_by_points[i].to_csv(df_fns[i])

    ds = open_mfcsv(df_fns, "id")

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

    ds = open_mfcsv(df_fns, "id")

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

    ds = open_mfcsv(fns, "id", segmented_by="var")

    assert sorted(list(ds.data_vars.keys())) == ["test1", "test2"], ds
    ids = ds.id.values
    for i in ids:
        test1 = ds.sel(id=i)["test1"]
        test2 = ds.sel(id=i)["test2"]
        assert np.all(np.equal(test1, np.arange(len(ids)) * int(i))), test1
        assert np.all(np.equal(test2, np.arange(len(ids)) ** int(i))), test2
