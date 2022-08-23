# -*- coding: utf-8 -*-
"""Tests for the hydromt.data_adapter submodule."""

import pytest
from os.path import join, dirname, abspath
import numpy as np
import geopandas as gpd
import pandas as pd
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
            for month in range(1, 13):
                with open(
                    join(
                        tmpdir, "{unknown_key}_" + f"{variable}_{year}_{month:02d}.nc"
                    ),
                    "w",
                ) as f:
                    f.write("")
    # create data catalog for these files
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "driver": "netcdf",
            "path": join(tmpdir, "{unknown_key}_{variable}_{year}_{month:02d}.nc"),
        }
    }
    cat = DataCatalog()
    cat.from_dict(dd)
    # test
    assert len(cat["test"].resolve_paths()) == 48
    assert len(cat["test"].resolve_paths(variables=["precip"])) == 24
    assert (
        len(
            cat["test"].resolve_paths(
                variables=["precip"], time_tuple=("2021-03-01", "2021-05-01")
            )
        )
        == 3
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


def test_dataframe(df, df_time, tmpdir):
    # Test reading csv
    fn_df = str(tmpdir.join("test.csv"))
    df.to_csv(fn_df)
    data_catalog = DataCatalog()
    df1 = data_catalog.get_dataframe(fn_df, index_col=0)
    assert isinstance(df1, pd.DataFrame)
    assert np.all(df1 == df)

    # Test FWF support
    fn_fwf = str(tmpdir.join("test.txt"))
    df.to_string(fn_fwf, index=False)
    fwf = data_catalog.get_dataframe(fn_fwf, driver="fwf", colspecs="infer")
    assert isinstance(fwf, pd.DataFrame)
    assert np.all(fwf == df)

    fn_xlsx = str(tmpdir.join("test.xlsx"))
    df.to_excel(fn_xlsx)
    df2 = data_catalog.get_dataframe(fn_xlsx, index_col=0)
    assert isinstance(df2, pd.DataFrame)
    assert np.all(df2 == df)


def test_dataframe_time(df_time, tmpdir):
    # Test time df
    fn_df_ts = str(tmpdir.join("test_ts.csv"))
    df_time.to_csv(fn_df_ts)
    data_catalog = DataCatalog()
    dfts1 = data_catalog.get_dataframe(fn_df_ts, index_col=0, parse_dates=True)
    assert isinstance(dfts1, pd.DataFrame)
    assert np.all(dfts1 == df_time)

    # Test renaming
    rename = {
        "precip": "P",
        "temp": "T",
        "pet": "ET",
    }
    dfts2 = data_catalog.get_dataframe(
        fn_df_ts, index_col=0, parse_dates=True, rename=rename
    )
    assert np.all(list(dfts2.columns) == list(rename.values()))

    # Test unit add/multiply
    unit_mult = {
        "precip": 0.75,
        "temp": 2,
        "pet": 1,
    }
    unit_add = {
        "precip": 0,
        "temp": -1,
        "pet": 2,
    }
    dfts3 = data_catalog.get_dataframe(
        fn_df_ts, index_col=0, parse_dates=True, unit_mult=unit_mult, unit_add=unit_add
    )
    # Do checks
    for var in df_time.columns:
        assert np.all(df_time[var] * unit_mult[var] + unit_add[var] == dfts3[var])

    # Test timeslice
    dfts4 = data_catalog.get_dataframe(
        fn_df_ts, index_col=0, parse_dates=True, time_tuple=("2007-01-02", "2007-01-04")
    )
    assert len(dfts4) == 3

    # Test variable slice
    vars_slice = ["precip", "temp"]
    dfts5 = data_catalog.get_dataframe(
        fn_df_ts, index_col=0, parse_dates=True, variables=vars_slice
    )
    assert np.all(dfts5.columns == vars_slice)
