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
from hydromt.data_adapter import DataAdapter, parse_data_sources, DataCatalog

TESTDATADIR = join(dirname(abspath(__file__)), "data")


def test_parse_sources():
    sources = parse_data_sources(join(TESTDATADIR, "test_sources.yml"))
    assert isinstance(sources, dict)
    assert np.all([isinstance(s, DataAdapter) for _, s in sources.items()])
    with pytest.raises(ValueError, match="Unknown type for path argument"):
        parse_data_sources(path=dict(test=False))
    with pytest.raises(ValueError, match="Unknown type for root argument"):
        parse_data_sources(path=TESTDATADIR, root=dict(test=False))
    with pytest.raises(FileNotFoundError):
        parse_data_sources("test_fail.yml")


def test_data_catalog_io(tmpdir):
    data_catalog = DataCatalog()
    # read / write
    fn_yml = str(tmpdir.join("test.yml"))
    data_catalog.to_yml(fn_yml)
    data_catalog1 = DataCatalog(data_libs=fn_yml)
    assert data_catalog.to_dict() == data_catalog1.to_dict()


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
    geoda.to_netcdf(fn_nc)
    geodf.to_file(fn_gdf, driver="GeoJSON")
    ts.to_csv(fn_csv)
    data_catalog = DataCatalog()
    # added fn_ts to test if it does not go into xr.open_dataset
    da1 = data_catalog.get_geodataset(
        fn_nc, rename={"test": "test1"}, fn_ts=None, bbox=geoda.vector.bounds
    ).sortby("index")
    assert np.allclose(da1, geoda) and da1.name == "test1"
    ds1 = data_catalog.get_geodataset("test", single_var_as_array=False)
    assert isinstance(ds1, xr.Dataset) and "test1" in ds1
    ds1 = data_catalog.get_geodataset(fn_gdf, fn_ts=fn_csv).sortby("index")
    assert np.allclose(da1, geoda)
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


def test_deltares_sources():
    data_catalog = DataCatalog(deltares_data=True)
    assert len(data_catalog._sources) > 0
    source0 = data_catalog._sources[[k for k in data_catalog.sources.keys()][0]]
    assert "wflow_global" in str(source0.path)


def test_artifact_sources():
    data_catalog = DataCatalog(artifact_data="v0.0.3")
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
