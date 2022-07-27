# -*- coding: utf-8 -*-
"""Tests for the hydromt.data_adapter submodule."""

import pytest
from os.path import join, dirname, abspath
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import os
import hydromt
from hydromt.data_adapter import (
    DataAdapter,
    RasterDatasetAdapter,
)
from hydromt.data_catalog import (
    DataCatalog,
    _parse_data_dict,
)


TESTDATADIR = join(dirname(abspath(__file__)), "data")


def test_parser():
    # valid abs root on windows and linux!
    root = "c:/root" if os.name == "nt" else "/c/root"
    # simple; abs path
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "path": f"{root}/to/data.tif",
        }
    }
    dd_out = _parse_data_dict(dd, root=root)
    assert isinstance(dd_out["test"], RasterDatasetAdapter)
    assert dd_out["test"].path == abspath(dd["test"]["path"])
    # rel path
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "path": "path/to/data.tif",
            "kwargs": {"fn": "test"},
        },
        "root": root,
    }
    dd_out = _parse_data_dict(dd)
    assert dd_out["test"].path == abspath(join(root, dd["test"]["path"]))
    # check if path in kwargs is also absolute
    assert dd_out["test"].kwargs["fn"] == abspath(join(root, "test"))
    # alias
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "path": "path/to/data.tif",
        },
        "test1": {"alias": "test"},
    }
    dd_out = _parse_data_dict(dd, root=root)
    assert dd_out["test"].path == dd_out["test1"].path
    # placeholder
    dd = {
        "test_{p1}_{p2}": {
            "data_type": "RasterDataset",
            "path": "data_{p2}.tif",
            "placeholders": {"p1": ["a", "b"], "p2": ["1", "2", "3"]},
        },
    }
    dd_out = _parse_data_dict(dd, root=root)
    assert len(dd_out) == 6
    assert dd_out["test_a_1"].path == abspath(join(root, "data_1.tif"))
    # errors
    with pytest.raises(ValueError, match="Missing required path argument"):
        _parse_data_dict({"test": {}})
    with pytest.raises(ValueError, match="Data type error unknown"):
        _parse_data_dict({"test": {"path": "", "data_type": "error"}})
    with pytest.raises(ValueError, match="alias test not found in data_dict"):
        _parse_data_dict({"test1": {"alias": "test"}})


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


def test_data_catalog_io(tmpdir):
    data_catalog = DataCatalog()
    # read / write
    fn_yml = str(tmpdir.join("test.yml"))
    data_catalog.to_yml(fn_yml)
    data_catalog1 = DataCatalog(data_libs=fn_yml)
    assert data_catalog.to_dict() == data_catalog1.to_dict()
    # test print
    print(data_catalog["merit_hydro"])


def test_data_catalog(tmpdir):
    data_catalog = DataCatalog()
    # initialized with empty dict
    assert len(data_catalog._sources) == 0
    # global data sources are automatically parsed
    assert len(data_catalog) > 0
    # test keys, getitem,
    keys = data_catalog.keys
    source = data_catalog[keys[0]]
    assert isinstance(source, DataAdapter)
    assert keys[0] in data_catalog
    # add source from dict
    data_dict = {keys[0]: source.to_dict()}
    data_catalog.from_dict(data_dict)
    # printers
    assert isinstance(data_catalog.__repr__(), str)
    assert isinstance(data_catalog._repr_html_(), str)
    assert isinstance(data_catalog.to_dataframe(), pd.DataFrame)
    with pytest.raises(ValueError, match="Value must be DataAdapter"):
        data_catalog["test"] = "string"


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
    df1 = data_catalog.get_dataframe(
        fn_df, index_col=0)
    assert isinstance(df1, pd.DataFrame)
    assert np.all(df1 == df)

    # Test FWF support
    fn_fwf = str(tmpdir.join("test.txt"))
    df.to_string(fn_fwf, index=False)
    fwf = data_catalog.get_dataframe(
        fn_fwf, driver="fwf", colspecs="infer")
    assert isinstance(fwf, pd.DataFrame)
    assert np.all(fwf == df)

def test_dataframe_time(df_time, tmpdir):
    # Test time df
    fn_df_ts = str(tmpdir.join("test_ts.csv"))
    df_time.to_csv(fn_df_ts)
    data_catalog = DataCatalog()
    dfts1 = data_catalog.get_dataframe(
        fn_df_ts, index_col=0, parse_dates=True)
    assert isinstance(dfts1, pd.DataFrame)
    assert np.all(dfts1 == df_time)

    # Test renaming
    rename = {
        "precip": "P", 
        "temp": "T", 
        "pet": "ET", 
    }
    dfts2 = data_catalog.get_dataframe(
        fn_df_ts, index_col=0, parse_dates=True, rename=rename)
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
        fn_df_ts, index_col=0, parse_dates=True, 
        unit_mult=unit_mult, unit_add=unit_add)
    # Do checks
    for var in df_time.columns:
        assert np.all(df_time[var] * unit_mult[var] + unit_add[var] == dfts3[var])

    # Test timeslice
    dfts4 = data_catalog.get_dataframe(
        fn_df_ts, index_col=0, parse_dates=True, 
        time_tuple=("2007-01-02", "2007-01-04"))
    assert len(dfts4) == 3

    # Test variable slice
    vars_slice = ["precip", "temp"]
    dfts5 = data_catalog.get_dataframe(
        fn_df_ts, index_col=0, parse_dates=True, 
        variables=vars_slice)
    assert np.all(dfts5.columns == vars_slice)

def test_deltares_sources():
    data_catalog = DataCatalog(deltares_data="v0.0.5")
    assert len(data_catalog._sources) > 0
    source0 = data_catalog._sources[[k for k in data_catalog.sources.keys()][0]]
    assert "wflow_global" in str(source0.path)


def test_artifact_sources():
    data_catalog = DataCatalog(artifact_data="v0.0.5")
    assert len(data_catalog._sources) > 0
    source0 = data_catalog._sources[[k for k in data_catalog.sources.keys()][0]]
    assert ".hydromt_data" in str(source0.path)


def test_export_global_datasets(tmpdir):
    DTYPES = {
        "RasterDatasetAdapter": (xr.DataArray, xr.Dataset),
        "GeoDatasetAdapter": (xr.DataArray, xr.Dataset),
        "GeoDataFrameAdapter": gpd.GeoDataFrame,
    }
    bbox = [11.70, 45.35, 12.95, 46.70]  # Piava river
    time_tuple = ("2010-02-01", "2010-02-14")
    data_catalog = DataCatalog()  # read artifacts by default
    sns = [
        "era5",
        "grwl_mask",
        "modis_lai",
        "osm_coastlines",
        "grdc",
        "corine",
        "gtsmv3_eu_era5",
        "hydro_lakes",
        "eobs",
    ]
    data_catalog.export_data(
        str(tmpdir), bbox=bbox, time_tuple=time_tuple, source_names=sns
    )

    data_catalog1 = DataCatalog(str(tmpdir.join("data_catalog.yml")))
    for key, source in data_catalog1.sources.items():
        source_type = type(source).__name__
        dtypes = DTYPES[source_type]
        obj = source.get_data()
        assert isinstance(obj, dtypes), key
