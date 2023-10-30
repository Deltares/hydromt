# -*- coding: utf-8 -*-
"""Tests for the hydromt.data_adapter submodule."""

import glob
import tempfile
from os.path import abspath, dirname, join
from typing import cast

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import hydromt
from hydromt import _compat as compat
from hydromt.data_adapter import (
    DatasetAdapter,
    GeoDataFrameAdapter,
    GeoDatasetAdapter,
    RasterDatasetAdapter,
)
from hydromt.data_catalog import DataCatalog
from hydromt.gis_utils import to_geographic_bbox

TESTDATADIR = join(dirname(abspath(__file__)), "data")
CATALOGDIR = join(dirname(abspath(__file__)), "..", "data", "catalogs")


def test_resolve_path(tmpdir):
    # create dummy files
    for variable in ["precip", "temp"]:
        for year in [2020, 2021]:
            for month in range(1, 13):
                fn = join(tmpdir, f"{{unknown_key}}_0_{variable}_{year}_{month:02d}.nc")
                with open(fn, "w") as f:
                    f.write("")
    # create data catalog for these files
    dd = {
        "test": {
            "data_type": "RasterDataset",
            "driver": "netcdf",
            "path": join(
                tmpdir, "{unknown_key}_{zoom_level}_{variable}_{year}_{month:02d}.nc"
            ),
        }
    }
    cat = DataCatalog()
    cat.from_dict(dd)
    source = cat.get_source("test")
    # test
    fns = source._resolve_paths()
    assert len(fns) == 48
    fns = source._resolve_paths(variables=["precip"])
    assert len(fns) == 24
    fns = source._resolve_paths(("2021-03-01", "2021-05-01"), ["precip"])
    assert len(fns) == 3
    with pytest.raises(FileNotFoundError, match="No such file found:"):
        source._resolve_paths(variables=["waves"])


def test_rasterdataset(rioda, tmpdir):
    fn_tif = str(tmpdir.join("test.tif"))
    rioda_utm = rioda.raster.reproject(dst_crs="utm")
    rioda_utm.raster.to_raster(fn_tif)
    data_catalog = DataCatalog()
    da1 = data_catalog.get_rasterdataset(fn_tif, bbox=rioda.raster.bounds)
    assert np.all(da1 == rioda_utm)
    geom = rioda.raster.box
    da1 = data_catalog.get_rasterdataset("test.tif", geom=geom)
    assert np.all(da1 == rioda_utm)
    with pytest.raises(FileNotFoundError, match="No such file or catalog source"):
        data_catalog.get_rasterdataset("no_file.tif")
    with pytest.raises(IndexError, match="RasterDataset: No data within"):
        data_catalog.get_rasterdataset("test.tif", bbox=[40, 50, 41, 51])


@pytest.mark.skipif(not compat.HAS_GCSFS, reason="GCSFS not installed.")
def test_gcs_cmip6(tmpdir):
    # TODO switch to pre-defined catalogs when pushed to main
    catalog_fn = join(CATALOGDIR, "gcs_cmip6_data.yml")
    data_catalog = DataCatalog(data_libs=[catalog_fn])
    ds = data_catalog.get_rasterdataset(
        "cmip6_NOAA-GFDL/GFDL-ESM4_historical_r1i1p1f1_Amon",
        variables=["precip", "temp"],
        time_tuple=(("1990-01-01", "1990-03-01")),
    )
    # Check reading and some preprocess
    assert "precip" in ds
    assert not np.any(ds[ds.raster.x_dim] > 180)
    # Skip as I don't think this adds value to testing a gcs cloud archive
    # Write and compare
    # fn_nc = str(tmpdir.join("test.nc"))
    # ds.to_netcdf(fn_nc)
    # ds1 = data_catalog.get_rasterdataset(fn_nc)
    # assert np.allclose(ds["precip"][0, :, :], ds1["precip"][0, :, :])


@pytest.mark.skipif(not compat.HAS_S3FS, reason="S3FS not installed.")
def test_aws_worldcover():
    catalog_fn = join(CATALOGDIR, "aws_data.yml")
    data_catalog = DataCatalog(data_libs=[catalog_fn])
    da = data_catalog.get_rasterdataset(
        "esa_worldcover_2020_v100",
        bbox=[12.0, 46.0, 12.5, 46.50],
    )
    assert da.name == "landuse"


def test_http_data():
    dc = DataCatalog().from_dict(
        {
            "global_wind_atlas": {
                "data_type": "RasterDataset",
                "driver": "raster",
                "path": "https://globalwindatlas.info/api/gis/global/wind-speed/10",
            }
        }
    )
    s = dc.get_source("global_wind_atlas")
    # test inferred file system
    assert isinstance(s.fs, fsspec.implementations.http.HTTPFileSystem)
    # test returns xarray DataArray
    da = s.get_data(bbox=[0, 0, 10, 10])
    assert isinstance(da, xr.DataArray)
    assert da.raster.shape == (4000, 4000)


def test_rasterdataset_zoomlevels(rioda_large, tmpdir):
    # write tif with zoom level 1 in name
    # NOTE zl 0 not written to check correct functioning
    name = "test_zoom"
    rioda_large.raster.to_raster(str(tmpdir.join("test_zl1.tif")))
    yml_dict = {
        name: {
            "crs": 4326,
            "data_type": "RasterDataset",
            "driver": "raster",
            "path": f"{str(tmpdir)}/test_zl{{zoom_level}}.tif",
            "zoom_levels": {0: 0.1, 1: 0.3},
        }
    }
    # test zoom levels in name
    data_catalog = DataCatalog()
    data_catalog.from_dict(yml_dict)
    rds = cast(RasterDatasetAdapter, data_catalog.get_source(name))
    assert rds._parse_zoom_level(None) is None
    assert rds._parse_zoom_level(zoom_level=1) == 1
    assert rds._parse_zoom_level(zoom_level=(0.3, "degree")) == 1
    assert rds._parse_zoom_level(zoom_level=(0.29, "degree")) == 0
    assert rds._parse_zoom_level(zoom_level=(0.1, "degree")) == 0
    assert rds._parse_zoom_level(zoom_level=(1, "meter")) == 0
    with pytest.raises(TypeError, match="zoom_level unit"):
        rds._parse_zoom_level(zoom_level=(1, "asfd"))
    with pytest.raises(TypeError, match="zoom_level not understood"):
        rds._parse_zoom_level(zoom_level=(1, "asfd", "asdf"))
    da1 = data_catalog.get_rasterdataset(name, zoom_level=(0.3, "degree"))
    assert isinstance(da1, xr.DataArray)
    # write COG
    cog_fn = str(tmpdir.join("test_cog.tif"))
    rioda_large.raster.to_raster(cog_fn, driver="COG", overviews="auto")
    # test COG zoom levels
    da1 = data_catalog.get_rasterdataset(cog_fn, zoom_level=(0.01, "degree"))
    assert da1.raster.shape == (256, 250)


def test_rasterdataset_driver_kwargs(artifact_data: DataCatalog, tmpdir):
    era5 = artifact_data.get_rasterdataset("era5")
    fp1 = join(tmpdir, "era5.zarr")
    era5.to_zarr(fp1)
    data_dict = {
        "era5_zarr": {
            "crs": 4326,
            "data_type": "RasterDataset",
            "driver": "zarr",
            "driver_kwargs": {
                "preprocess": "round_latlon",
            },
            "path": fp1,
        }
    }
    datacatalog = DataCatalog()
    datacatalog.from_dict(data_dict)
    era5_zarr = datacatalog.get_rasterdataset("era5_zarr")
    fp2 = join(tmpdir, "era5.nc")
    era5.to_netcdf(fp2)

    data_dict2 = {
        "era5_nc": {
            "crs": 4326,
            "data_type": "RasterDataset",
            "driver": "netcdf",
            "driver_kwargs": {
                "preprocess": "round_latlon",
            },
            "path": fp2,
        }
    }
    datacatalog.from_dict(data_dict2)
    era5_nc = datacatalog.get_rasterdataset("era5_nc")
    assert era5_zarr.equals(era5_nc)
    datacatalog.get_source("era5_zarr").to_file(tmpdir, "era5_zarr", driver="zarr")


def test_rasterdataset_unit_attrs(artifact_data: DataCatalog):
    era5_dict = {"era5": artifact_data.get_source("era5").to_dict()}
    attrs = {
        "temp": {"unit": "degrees C", "long_name": "temperature"},
        "temp_max": {"unit": "degrees C", "long_name": "maximum temperature"},
        "temp_min": {"unit": "degrees C", "long_name": "minimum temperature"},
    }
    era5_dict["era5"].update(dict(attrs=attrs))
    artifact_data.from_dict(era5_dict)
    raster = artifact_data.get_rasterdataset("era5")
    assert raster["temp"].attrs["unit"] == attrs["temp"]["unit"]
    assert raster["temp_max"].attrs["long_name"] == attrs["temp_max"]["long_name"]


# @pytest.mark.skip()
def test_geodataset(geoda, geodf, ts, tmpdir):
    fn_nc = str(tmpdir.join("test.nc"))
    fn_gdf = str(tmpdir.join("test.geojson"))
    fn_csv = str(tmpdir.join("test.csv"))
    fn_csv_locs = str(tmpdir.join("test_locs.xy"))
    geoda.vector.to_netcdf(fn_nc)
    geodf.to_file(fn_gdf, driver="GeoJSON")
    ts.to_csv(fn_csv)
    hydromt.io.write_xy(fn_csv_locs, geodf)
    data_catalog = DataCatalog()
    # added fn_ts to test if it does not go into xr.open_dataset
    da1 = data_catalog.get_geodataset(
        fn_nc, variables=["test1"], bbox=geoda.vector.bounds
    ).sortby("index")
    assert np.allclose(da1, geoda)
    assert da1.name == "test1"
    ds1 = data_catalog.get_geodataset("test.nc", single_var_as_array=False)
    assert isinstance(ds1, xr.Dataset)
    assert "test" in ds1
    da2 = data_catalog.get_geodataset(
        fn_gdf, driver_kwargs=dict(fn_data=fn_csv)
    ).sortby("index")
    assert isinstance(da2, xr.DataArray), type(da2)
    assert np.allclose(da2, geoda)
    # test with xy locs
    da3 = data_catalog.get_geodataset(
        fn_csv_locs, driver_kwargs=dict(fn_data=fn_csv), crs=geodf.crs
    ).sortby("index")
    assert np.allclose(da3, geoda)
    assert da3.vector.crs.to_epsg() == 4326
    with pytest.raises(FileNotFoundError, match="No such file or catalog source"):
        data_catalog.get_geodataset("no_file.geojson")
    with tempfile.TemporaryDirectory() as td:
        # Test nc file writing to file
        GeoDatasetAdapter(fn_nc).to_file(
            data_root=td, data_name="test", driver="netcdf"
        )
        GeoDatasetAdapter(fn_nc).to_file(
            data_root=tmpdir, data_name="test1", driver="netcdf", variables="test1"
        )
        GeoDatasetAdapter(fn_nc).to_file(data_root=td, data_name="test", driver="zarr")


def test_geodataset_unit_attrs(artifact_data: DataCatalog):
    gtsm_dict = {"gtsmv3_eu_era5": artifact_data.get_source("gtsmv3_eu_era5").to_dict()}
    attrs = {
        "waterlevel": {
            "long_name": "sea surface height above mean sea level",
            "unit": "meters",
        }
    }
    gtsm_dict["gtsmv3_eu_era5"].update(dict(attrs=attrs))
    artifact_data.from_dict(gtsm_dict)
    gtsm_geodataarray = artifact_data.get_geodataset("gtsmv3_eu_era5")
    assert gtsm_geodataarray.attrs["long_name"] == attrs["waterlevel"]["long_name"]
    assert gtsm_geodataarray.attrs["unit"] == attrs["waterlevel"]["unit"]


def test_geodataset_unit_conversion(artifact_data: DataCatalog):
    gtsm_geodataarray = artifact_data.get_geodataset("gtsmv3_eu_era5")
    gtsm_dict = {"gtsmv3_eu_era5": artifact_data.get_source("gtsmv3_eu_era5").to_dict()}
    gtsm_dict["gtsmv3_eu_era5"].update(dict(unit_mult=dict(waterlevel=1000)))
    datacatalog = DataCatalog()
    datacatalog.from_dict(gtsm_dict)
    gtsm_geodataarray1000 = datacatalog.get_geodataset("gtsmv3_eu_era5")
    assert gtsm_geodataarray1000.equals(gtsm_geodataarray * 1000)


def test_geodataset_set_nodata(artifact_data: DataCatalog):
    gtsm_dict = {"gtsmv3_eu_era5": artifact_data.get_source("gtsmv3_eu_era5").to_dict()}
    gtsm_dict["gtsmv3_eu_era5"].update(dict(nodata=-99))
    datacatalog = DataCatalog()
    datacatalog.from_dict(gtsm_dict)
    ds = datacatalog.get_geodataset("gtsmv3_eu_era5")
    assert ds.vector.nodata == -99


def test_dataset_get_data(timeseries_df, tmpdir):
    ds = timeseries_df.to_xarray()
    path = str(tmpdir.join("test.nc"))
    ds.to_netcdf(path)
    dataset_adapter = DatasetAdapter(path=path, driver="netcdf")
    ds1 = dataset_adapter.get_data()
    assert isinstance(ds1, xr.Dataset)
    assert ds1.identical(ds)


def test_dataset_to_file(timeseries_df, tmpdir):
    ds = timeseries_df[["col1", "col2"]].to_xarray()
    path = str(tmpdir.join("test1.nc"))
    encoding = {k: {"zlib": True} for k in ds.vector.vars}
    ds.to_netcdf(path, encoding=encoding)
    dataset_adapter = DatasetAdapter(path=path, driver="netcdf")
    fn_out, driver = dataset_adapter.to_file(
        data_root=tmpdir, data_name="test2", driver="netcdf"
    )
    assert driver == "netcdf"
    assert fn_out == str(tmpdir.join("test2.nc"))
    ds2 = xr.open_dataset(fn_out)
    assert ds2.identical(ds)

    variables = ["col1", "col2"]
    fn_out, driver = dataset_adapter.to_file(
        data_root=tmpdir, data_name="test3", driver="netcdf", variables=variables
    )
    for variable in variables:
        ds3 = xr.open_dataset(fn_out.format(variable=variable))
        assert variable in ds3.vector.vars

    fn_out, driver = dataset_adapter.to_file(
        data_root=tmpdir, data_name="test4", driver="zarr"
    )

    ds_zarr = xr.open_zarr(fn_out)
    assert ds_zarr.identical(ds)

    with pytest.raises(ValueError, match="Dataset: Driver fake-driver unknown."):
        dataset_adapter.to_file(
            data_root=tmpdir, data_name="test4", driver="fake-driver"
        )


def test_dataset__read_data(tmpdir, timeseries_df):
    ds = timeseries_df[["col1", "col2"]].to_xarray()
    zarr_path = str(tmpdir.join("zarr_data"))

    ds.to_zarr(zarr_path)
    dataset_adapter = DatasetAdapter(path=zarr_path, driver="zarr")
    dataset_adapter.get_data(variables=["col1", "col2"])

    dataset_adapter = DatasetAdapter(path=zarr_path, driver="fake-driver")
    with pytest.raises(ValueError, match=r"Dataset: Driver fake-driver unknown"):
        dataset_adapter.get_data(variables=["col1", "col2"])


def test_dataset__set_nodata(tmpdir, timeseries_df):
    ds = timeseries_df[["col1", "col2"]].to_xarray()
    path = str(tmpdir.join("test.nc"))
    ds.to_netcdf(path)

    dataset_adapter = DatasetAdapter(
        path=path, driver="netcdf", nodata=dict(nodata=-99)
    )
    ds = dataset_adapter.get_data()
    raise NotImplementedError


def test_geodataframe(geodf, tmpdir):
    fn_gdf = str(tmpdir.join("test.geojson"))
    geodf.to_file(fn_gdf, driver="GeoJSON")
    data_catalog = DataCatalog()
    gdf1 = data_catalog.get_geodataframe(fn_gdf, bbox=geodf.total_bounds)
    assert isinstance(gdf1, gpd.GeoDataFrame)
    assert np.all(gdf1 == geodf)
    gdf1 = data_catalog.get_geodataframe(
        "test.geojson", bbox=geodf.total_bounds, buffer=1000, rename={"test": "test1"}
    )
    assert np.all(gdf1 == geodf)
    with pytest.raises(FileNotFoundError, match="No such file or catalog source"):
        data_catalog.get_geodataframe("no_file.geojson")


def test_geodataframe_unit_attrs(artifact_data: DataCatalog):
    gadm_level1 = {"gadm_level1": artifact_data.get_source("gadm_level1").to_dict()}
    attrs = {"NAME_0": {"long_name": "Country names"}}
    gadm_level1["gadm_level1"].update(dict(attrs=attrs))
    artifact_data.from_dict(gadm_level1)
    gadm_level1_gdf = artifact_data.get_geodataframe("gadm_level1")
    assert gadm_level1_gdf["NAME_0"].attrs["long_name"] == "Country names"


def test_dataframe(df, tmpdir):
    # Test reading csv
    fn_df = str(tmpdir.join("test.csv"))
    df.to_csv(fn_df)
    data_catalog = DataCatalog()
    df1 = data_catalog.get_dataframe(fn_df, driver_kwargs=dict(index_col=0))
    assert isinstance(df1, pd.DataFrame)
    pd.testing.assert_frame_equal(df, df1)

    # test reading parquet
    fn_df_parquet = str(tmpdir.join("test.parquet"))
    df.to_parquet(fn_df_parquet)
    data_catalog = DataCatalog()
    df2 = data_catalog.get_dataframe(fn_df_parquet, driver="parquet")
    assert isinstance(df2, pd.DataFrame)
    pd.testing.assert_frame_equal(df, df2)

    # Test FWF support
    fn_fwf = str(tmpdir.join("test.txt"))
    df.to_string(fn_fwf, index=False)
    fwf = data_catalog.get_dataframe(
        fn_fwf, driver="fwf", driver_kwargs=dict(colspecs="infer")
    )
    assert isinstance(fwf, pd.DataFrame)
    assert np.all(fwf == df)

    if compat.HAS_OPENPYXL:
        fn_xlsx = str(tmpdir.join("test.xlsx"))
        df.to_excel(fn_xlsx)
        df3 = data_catalog.get_dataframe(fn_xlsx, driver_kwargs=dict(index_col=0))
        assert isinstance(df3, pd.DataFrame)
        assert np.all(df3 == df)


def test_dataframe_unit_attrs(df: pd.DataFrame, tmpdir):
    df_path = join(tmpdir, "cities.csv")
    df["test_na"] = -9999
    df.to_csv(df_path)
    cities = {
        "cities": {
            "path": df_path,
            "data_type": "DataFrame",
            "driver": "csv",
            "nodata": -9999,
            "attrs": {
                "city": {"long_name": "names of cities"},
                "country": {"long_name": "names of countries"},
            },
        }
    }
    datacatalog = DataCatalog()
    datacatalog.from_dict(cities)
    cities_df = datacatalog.get_dataframe("cities")
    assert cities_df["city"].attrs["long_name"] == "names of cities"
    assert cities_df["country"].attrs["long_name"] == "names of countries"
    assert np.all(cities_df["test_na"].isna())


def test_dataframe_time(df_time, tmpdir):
    # Test time df
    fn_df_ts = str(tmpdir.join("test_ts.csv"))
    df_time.to_csv(fn_df_ts)
    data_catalog = DataCatalog()
    dfts1 = data_catalog.get_dataframe(
        fn_df_ts, driver_kwargs=dict(index_col=0, parse_dates=True)
    )
    assert isinstance(dfts1, pd.DataFrame)
    assert np.all(dfts1 == df_time)

    # Test renaming
    rename = {
        "precip": "P",
        "temp": "T",
        "pet": "ET",
    }
    dfts2 = data_catalog.get_dataframe(
        fn_df_ts, driver_kwargs=dict(index_col=0, parse_dates=True), rename=rename
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
        fn_df_ts,
        driver_kwargs=dict(index_col=0, parse_dates=True),
        unit_mult=unit_mult,
        unit_add=unit_add,
    )
    # Do checks
    for var in df_time.columns:
        assert np.all(df_time[var] * unit_mult[var] + unit_add[var] == dfts3[var])

    # Test timeslice
    dfts4 = data_catalog.get_dataframe(
        fn_df_ts,
        time_tuple=("2007-01-02", "2007-01-04"),
        driver_kwargs=dict(index_col=0, parse_dates=True),
    )
    assert len(dfts4) == 3

    # Test variable slice
    vars_slice = ["precip", "temp"]
    dfts5 = data_catalog.get_dataframe(
        fn_df_ts,
        variables=vars_slice,
        driver_kwargs=dict(index_col=0, parse_dates=True),
    )
    assert np.all(dfts5.columns == vars_slice)


def test_cache_vrt(tmpdir, rioda_large):
    # write vrt data
    name = "tiled"
    root = str(tmpdir.join(name))
    rioda_large.raster.to_xyz_tiles(
        root=root,
        tile_size=256,
        zoom_levels=[0],
    )
    cat = DataCatalog(join(root, f"{name}.yml"), cache=True)
    cat.get_rasterdataset(name)
    assert len(glob.glob(join(cat._cache_dir, name, name, "*", "*", "*.tif"))) == 16


def test_detect_extent(geodf, geoda, rioda, ts):
    ts_expected_bbox = (-74.08, -34.58, -47.91, 10.48)
    ts_detected_bbox = to_geographic_bbox(*GeoDataFrameAdapter("").detect_bbox(geodf))
    assert np.all(np.equal(ts_expected_bbox, ts_detected_bbox))

    geoda_expected_time_range = tuple(pd.to_datetime(["01-01-2000", "12-31-2000"]))
    geoda_expected_bbox = (-74.08, -34.58, -47.91, 10.48)
    geoda_detected_bbox = to_geographic_bbox(*GeoDatasetAdapter("").detect_bbox(geoda))
    geoda_detected_time_range = GeoDatasetAdapter("").detect_time_range(geoda)
    assert np.all(np.equal(geoda_expected_bbox, geoda_detected_bbox))
    assert geoda_expected_time_range == geoda_detected_time_range

    rioda_expected_bbox = (3.0, -11.0, 6.0, -9.0)
    rioda_detected_bbox = to_geographic_bbox(
        *RasterDatasetAdapter("").detect_bbox(rioda)
    )

    assert np.all(np.equal(rioda_expected_bbox, rioda_detected_bbox))
