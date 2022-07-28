# -*- coding: utf-8 -*-
"""Tests for the hydromt.data_adapter submodule."""

import pytest
from os.path import join, dirname, abspath
import numpy as np
import geopandas as gpd
import xarray as xr
import hydromt

from hydromt.data_catalog import (
    DataCatalog,
)

TESTDATADIR = join(dirname(abspath(__file__)), "data")


def test_resolve_path(tmpdir):
    # create dummy files
    for variable in ["precip", "temp"]:
        for year in [2020, 2021]:
            with open(
                join(tmpdir, "{unknown_key}_" + f"{variable}_{year}.nc"), "w"
            ) as f:
                f.write("")
    # create data catalog for these files
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "driver": "netcdf",
            "path": join(tmpdir, "{unknown_key}_{variable}_{year}.nc"),
        }
    }
    cat = DataCatalog()
    cat.from_dict(dd)
    # test
    assert len(cat["test"].resolve_paths()) == 4
    assert len(cat["test"].resolve_paths(variables=["precip"])) == 2
    assert (
        len(
            cat["test"].resolve_paths(
                variables=["precip"], time_tuple=("2021-03-01", "2021-05-01")
            )
        )
        == 1
    )
    with pytest.raises(FileNotFoundError, match="No such file found:"):
        cat["test"].resolve_paths(variables=["waves"])


def test_rasterdataset(rioda, tmpdir):
    fn_tif = str(tmpdir.join("test.tif"))
    rioda.raster.to_raster(fn_tif)
    data_catalog = DataCatalog()
    da1 = data_catalog.get_rasterdataset(fn_tif, bbox=rioda.raster.bounds)
    assert np.all(da1 == rioda)
    da1 = data_catalog.get_rasterdataset("test", geom=rioda.raster.box)
    assert np.all(da1 == rioda)
    with pytest.raises(FileNotFoundError, match="No such file or catalog key"):
        data_catalog.get_rasterdataset("no_file.tif")


def test_geodataset(geoda, geodf, ts, tmpdir):
    fn_nc = str(tmpdir.join("test.nc"))
    fn_gdf = str(tmpdir.join("test.geojson"))
    fn_csv = str(tmpdir.join("test.csv"))
    fn_csv_locs = str(tmpdir.join("test_locs.xy"))
    geoda.to_netcdf(fn_nc)
    geodf.to_file(fn_gdf, driver="GeoJSON")
    ts.to_csv(fn_csv)
    hydromt.io.write_xy(fn_csv_locs, geodf)
    data_catalog = DataCatalog()
    # added fn_ts to test if it does not go into xr.open_dataset
    da1 = data_catalog.get_geodataset(
        fn_nc, variables=["test1"], bbox=geoda.vector.bounds
    ).sortby("index")
    assert np.allclose(da1, geoda) and da1.name == "test1"
    ds1 = data_catalog.get_geodataset("test", single_var_as_array=False)
    assert isinstance(ds1, xr.Dataset) and "test" in ds1
    da2 = data_catalog.get_geodataset(fn_gdf, fn_data=fn_csv).sortby("index")
    assert np.allclose(da2, geoda)
    # test with xy locs
    da3 = data_catalog.get_geodataset(
        fn_csv_locs, fn_data=fn_csv, crs=geodf.crs
    ).sortby("index")
    assert np.allclose(da3, geoda)
    assert da3.vector.crs.to_epsg() == 4326
    with pytest.raises(FileNotFoundError, match="No such file or catalog key"):
        data_catalog.get_geodataset("no_file.geojson")


def test_geodataframe(geodf, tmpdir):
    fn_gdf = str(tmpdir.join("test.geojson"))
    geodf.to_file(fn_gdf, driver="GeoJSON")
    data_catalog = DataCatalog()
    gdf1 = data_catalog.get_geodataframe(fn_gdf, bbox=geodf.total_bounds)
    assert isinstance(gdf1, gpd.GeoDataFrame)
    assert np.all(gdf1 == geodf)
    gdf1 = data_catalog.get_geodataframe(
        "test", bbox=geodf.total_bounds, buffer=1000, rename={"test": "test1"}
    )
    assert np.all(gdf1 == geodf)
    with pytest.raises(FileNotFoundError, match="No such file or catalog key"):
        data_catalog.get_geodataframe("no_file.geojson")
