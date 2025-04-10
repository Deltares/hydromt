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
from hydromt._io.readers import (
    _open_geodataset,
    _open_mfcsv,
    _open_timeseries_from_table,
    _open_vector,
    _open_vector_from_table,
    _read_toml,
)
from hydromt._io.writers import _write_toml, _write_xy


def test_open_vector(tmpdir, df, geodf, world):
    csv_path = str(tmpdir.join("test.csv"))
    parquet_path = str(tmpdir.join("test.parquet"))
    xy_path = str(tmpdir.join("test.xy"))
    xls_path = str(tmpdir.join("test.xlsx"))
    geojson_path = str(tmpdir.join("test.geojson"))
    shp_path = str(tmpdir.join("test.shp"))
    gpkg_path = str(tmpdir.join("test.gpkg"))
    df.to_csv(csv_path)
    df.to_parquet(parquet_path)
    if _compat.HAS_OPENPYXL:
        df.to_excel(xls_path)
    geodf.to_file(geojson_path, driver="GeoJSON")
    _write_xy(xy_path, geodf)
    geodf.to_file(shp_path)
    geodf.to_file(gpkg_path, driver="GPKG")
    # read csv
    gdf1 = _open_vector(csv_path, assert_gtype="Point", crs=4326)
    assert gdf1.crs == geodf.crs
    assert np.all(gdf1 == geodf)
    # no data in domain
    gdf1 = _open_vector(csv_path, crs=4326, bbox=[200, 300, 200, 300])
    assert gdf1.index.size == 0
    if _compat.HAS_OPENPYXL:
        # read xls
        gdf1 = _open_vector(xls_path, assert_gtype="Point", crs=4326)
        assert np.all(gdf1 == geodf)

    # read xy
    gdf1 = _open_vector(xy_path, crs=4326)
    assert np.all(gdf1 == geodf[["geometry"]])
    # read shapefile
    gdf1 = hydromt._io._open_vector(shp_path, bbox=list(geodf.total_bounds))
    assert np.all(gdf1 == geodf)
    mask = gpd.GeoDataFrame({"geometry": [box(*geodf.total_bounds)]}, crs=geodf.crs)
    gdf1 = hydromt._io._open_vector(shp_path, geom=mask)
    assert np.all(gdf1 == geodf)
    # read geopackage
    gdf1 = hydromt._io._open_vector(gpkg_path)
    assert np.all(gdf1 == geodf)
    # read parquet
    gdf1 = _open_vector(parquet_path, crs=4326)
    assert np.all(gdf1 == geodf)
    # filter
    country = "Chile"
    geom = world[world["name"] == country]
    gdf1 = _open_vector(
        csv_path, crs=4326, geom=geom.to_crs(3857)
    )  # crs should default to 4326
    assert np.all(gdf1["country"] == country)
    gdf2 = _open_vector(geojson_path, geom=geom)
    # NOTE labels are different
    assert np.all(gdf1.geometry.values == gdf2.geometry.values)
    gdf2 = _open_vector(csv_path, crs=4326, bbox=geom.total_bounds)
    assert np.all(gdf1.geometry.values == gdf2.geometry.values)
    # error
    with pytest.raises(ValueError, match="other geometries"):
        _open_vector(csv_path, assert_gtype="Polygon")
    with pytest.raises(ValueError, match="unknown"):
        _open_vector(csv_path, assert_gtype="PolygonPoints")
    with pytest.raises(ValueError, match="The GeoDataFrame has no CRS"):
        _open_vector(csv_path)
    with pytest.raises(ValueError, match="Unknown geometry mask type"):
        _open_vector(csv_path, crs=4326, geom=geom.total_bounds)
    with pytest.raises(ValueError, match="x dimension"):
        _open_vector(csv_path, x_dim="x")
    with pytest.raises(ValueError, match="y dimension"):
        _open_vector(csv_path, y_dim="y")
    with pytest.raises(IOError, match="No such file"):
        _open_vector("fail.csv")
    with pytest.raises(IOError, match="Driver fail unknown"):
        _open_vector_from_table("test.fail")


@pytest.mark.skipif(not _compat.HAS_S3FS, reason="S3FS not installed.")
def test_open_vector_s3(geodf: gpd.GeoDataFrame):
    m = MagicMock()
    m.return_value = geodf
    with patch("geopandas.io.file._read_file_pyogrio", m):
        df = hydromt._io._open_vector(UPath("s3://fake_url/file.geojson"))
    assert np.all(geodf == df)


def test_open_geodataset(tmpdir, geodf):
    point_data_path = str(tmpdir.join("points.geojson"))
    geodf.to_file(point_data_path, driver="GeoJSON")
    # create equivalent polygon file
    polygon_data_path = str(tmpdir.join("polygons.geojson"))
    geodf_poly = geodf.copy()
    crs = geodf.crs
    geodf_poly["geometry"] = geodf_poly.to_crs(3857).buffer(0.1).to_crs(crs)
    geodf_poly.to_file(polygon_data_path, driver="GeoJSON")
    # create zeros timeseries
    ts = pd.DataFrame(
        index=pd.DatetimeIndex(["01-01-2000", "01-01-2001"]),
        columns=geodf.index.values,
        data=np.zeros((2, geodf.index.size)),
    )
    name = "waterlevel"
    timeseries_path = str(tmpdir.join(f"{name}.csv"))
    ts.to_csv(timeseries_path)
    # returns dataset with coordinates, but no variable
    ds = _open_geodataset(point_data_path)
    assert isinstance(ds, xr.Dataset)
    assert len(ds.data_vars) == 0
    geodf1 = ds.vector.to_gdf()
    assert np.all(geodf == geodf1[geodf.columns])
    # add timeseries
    ds = _open_geodataset(point_data_path, data_path=timeseries_path)
    assert name in ds.data_vars
    assert np.all(ds[name].values == 0)
    # test for polygon geometry
    ds = _open_geodataset(polygon_data_path, data_path=timeseries_path)
    assert name in ds.data_vars
    assert ds.vector.geom_type == "Polygon"
    with pytest.raises(IOError, match="GeoDataset point location file not found"):
        _open_geodataset("missing_file.csv")
    with pytest.raises(IOError, match="GeoDataset data file not found"):
        _open_geodataset(point_data_path, data_path="missing_file.csv")


def test_timeseries_io(tmpdir, ts):
    ts_path = str(tmpdir.join("test1.csv"))
    # dattime in columns
    ts.to_csv(ts_path)
    da = _open_timeseries_from_table(ts_path)
    assert isinstance(da, xr.DataArray)
    assert da.time.dtype.type.__name__ == "datetime64"
    # transposed df > datetime in row index
    ts_transposed_path = str(tmpdir.join("test2.csv"))
    ts = ts.T
    ts.to_csv(ts_transposed_path)
    da2 = _open_timeseries_from_table(ts_transposed_path)
    assert da.time.dtype.type.__name__ == "datetime64"
    assert np.all(da == da2)
    # no time index
    ts_no_index_path = str(tmpdir.join("test3.csv"))
    pd.DataFrame(ts.values).to_csv(ts_no_index_path)
    with pytest.raises(ValueError, match="No time index found"):
        _open_timeseries_from_table(ts_no_index_path)
    # parse str index to numeric index
    cols = [f"a_{i}" for i in ts.columns]
    ts.columns = cols
    ts_num_index_path = str(tmpdir.join("test4.csv"))
    ts.to_csv(ts_num_index_path)
    da4 = _open_timeseries_from_table(ts_num_index_path)
    assert np.all(da == da4)
    assert np.all(da.index == da4.index)
    # no numeric index
    cols[0] = "a"
    ts.columns = cols
    ts_no_num_index_path = str(tmpdir.join("test5.csv"))
    ts.to_csv(ts_no_num_index_path)
    with pytest.raises(ValueError, match="No numeric index"):
        _open_timeseries_from_table(ts_no_num_index_path)


def test_open_mfcsv_by_id(tmpdir, dfs_segmented_by_points):
    df_paths = {
        i: str(tmpdir.join("data", f"{i}.csv"))
        for i in range(len(dfs_segmented_by_points))
    }
    os.mkdir(tmpdir.join("data"))
    for i in range(len(df_paths)):
        dfs_segmented_by_points[i].to_csv(df_paths[i])

    ds = _open_mfcsv(df_paths, "id")

    assert sorted(list(ds.data_vars.keys())) == ["test1", "test2"], ds
    assert sorted(list(ds.dims)) == ["id", "time"], ds
    for i in range(len(dfs_segmented_by_points)):
        test1 = ds.sel(id=i)["test1"]
        test2 = ds.sel(id=i)["test2"]
        assert np.all(np.equal(test1, np.arange(len(dfs_segmented_by_points)) * i)), (
            test1
        )
        assert np.all(np.equal(test2, np.arange(len(dfs_segmented_by_points)) ** i)), (
            test2
        )

    # again but with a nameless csv index
    for i in range(len(df_paths)):
        dfs_segmented_by_points[i].rename_axis(None, axis=0, inplace=True)
        dfs_segmented_by_points[i].to_csv(df_paths[i])

    ds = _open_mfcsv(df_paths, "id")

    assert sorted(list(ds.data_vars.keys())) == ["test1", "test2"], ds
    assert sorted(list(ds.dims)) == ["id", "index"], ds
    for i in range(len(dfs_segmented_by_points)):
        test1 = ds.sel(id=i)["test1"]
        test2 = ds.sel(id=i)["test2"]
        assert np.all(np.equal(test1, np.arange(len(dfs_segmented_by_points)) * i)), (
            test1
        )
        assert np.all(np.equal(test2, np.arange(len(dfs_segmented_by_points)) ** i)), (
            test2
        )


def test_open_mfcsv_by_var(tmpdir, dfs_segmented_by_vars):
    os.mkdir(tmpdir.join("data"))
    paths = {}
    for var, df in dfs_segmented_by_vars.items():
        csv_path = tmpdir.join("data", f"{var}.csv")
        df.to_csv(csv_path)
        paths[var] = csv_path

    ds = _open_mfcsv(paths, "id", segmented_by="var")

    assert sorted(list(ds.data_vars.keys())) == ["test1", "test2"], ds
    ids = ds.id.values
    for i in ids:
        test1 = ds.sel(id=i)["test1"]
        test2 = ds.sel(id=i)["test2"]
        assert np.all(np.equal(test1, np.arange(len(ids)) * int(i))), test1
        assert np.all(np.equal(test2, np.arange(len(ids)) ** int(i))), test2


def test_toml_io(tmp_path):
    toml_content = """
    # This is an input section of a wflow toml config file

    [input]
    path_forcing = "inmaps.nc"
    path_static = "staticmaps.nc"
    ldd = "wflow_ldd"
    river_location = "wflow_river"
    subcatchment = "wflow_subcatch"
    forcing = [ "vertical.precipitation", "vertical.temperature", "vertical.potential_evaporation",]
    cyclic = [ "vertical.leaf_area_index",]
    gauges = "wflow_gauges"
    gauges_grdc = "wflow_gauges_grdc"

    """
    test_toml_fp = tmp_path / "test.toml"
    with open(test_toml_fp, "w") as f:
        f.write(toml_content)

    # Read toml
    config = _read_toml(test_toml_fp)

    # Check if structure and comment is preserverd
    assert config.as_string() == toml_content

    # test write
    _write_toml(test_toml_fp, config)
    config2 = _read_toml(test_toml_fp)
    assert config2 == config
