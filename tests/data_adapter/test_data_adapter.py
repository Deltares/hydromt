# -*- coding: utf-8 -*-
"""Tests for the hydromt.data_adapter submodule."""

import glob
import tempfile
from datetime import datetime
from os.path import abspath, basename, dirname, join
from pathlib import Path
from typing import cast

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem
from shapely import box

import hydromt
from hydromt import _compat as compat
from hydromt._typing import ErrorHandleMethod
from hydromt._typing.error import NoDataException, NoDataStrategy
from hydromt.data_adapter import (
    DatasetAdapter,
    GeoDataFrameAdapter,
    GeoDatasetAdapter,
    RasterDatasetAdapter,
)
from hydromt.data_catalog import DataCatalog
from hydromt.data_source import DataSource, RasterDatasetSource
from hydromt.gis.utils import to_geographic_bbox
from hydromt.io.writers import write_xy

TESTDATADIR = join(dirname(abspath(__file__)), "..", "data")
CATALOGDIR = join(dirname(abspath(__file__)), "..", "..", "data", "catalogs")


@pytest.mark.skip(reason="Needs refactor from path to uri.")
@pytest.mark.skipif(not compat.HAS_GCSFS, reason="GCSFS not installed.")
def test_gcs_cmip6():
    # TODO switch to pre-defined catalogs when pushed to main
    catalog_fn = join(CATALOGDIR, "gcs_cmip6_data", "v0.1.0", "data_catalog.yml")
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


@pytest.mark.skip(reason="Needs implementation of all Drivers.")
@pytest.mark.skipif(not compat.HAS_S3FS, reason="S3FS not installed.")
def test_aws_worldcover():
    catalog_fn = join(CATALOGDIR, "aws_data", "v0.1.0", "data_catalog.yml")
    data_catalog = DataCatalog(data_libs=[catalog_fn])
    da = data_catalog.get_rasterdataset(
        "esa_worldcover_2020_v100",
        bbox=[12.0, 46.0, 12.5, 46.50],
    )
    assert da.name == "landuse"


@pytest.mark.integration()
def test_http_data():
    dc = DataCatalog().from_dict(
        {
            "global_wind_atlas": {
                "data_type": "RasterDataset",
                "driver": {"name": "rasterio", "filesystem": "http"},
                "uri": "https://globalwindatlas.info/api/gis/global/wind-speed/10",
            }
        }
    )
    s: DataSource = dc.get_source("global_wind_atlas")
    # test inferred file system
    assert isinstance(s.driver.filesystem, fsspec.implementations.http.HTTPFileSystem)
    # test returns xarray DataArray
    da = s.read_data(bbox=[0, 0, 10, 10])
    assert isinstance(da, xr.DataArray)
    assert da.raster.shape == (4000, 4000)


@pytest.mark.skip(
    "Needs implementation of https://github.com/Deltares/hydromt/issues/875"
)
@pytest.mark.integration()
def test_rasterdataset_zoomlevels(
    rioda_large: xr.DataArray, tmp_dir: Path, artifact_data_catalog: DataCatalog
):
    # write tif with zoom level 1 in name
    # NOTE zl 0 not written to check correct functioning
    name = "test_zoom"
    rioda_large.raster.to_raster(str(tmp_dir / "test_zl1.tif"))
    yml_dict = {
        name: {
            "data_type": "RasterDataset",
            "driver": {"name": "rasterio"},
            "uri": f"{str(tmp_dir)}/test_zl{{zoom_level:d}}.tif",  # test with str format for zoom level
            "metadata": {
                "crs": 4326,
            },
            "zoom_levels": {0: 0.1, 1: 0.3},
        }
    }
    # test zoom levels in name
    artifact_data_catalog.from_dict(yml_dict)
    rds: RasterDatasetSource = cast(
        RasterDatasetSource, artifact_data_catalog.get_source(name)
    )
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
    da1 = artifact_data_catalog.get_rasterdataset(name, zoom_level=(0.3, "degree"))
    assert isinstance(da1, xr.DataArray)
    # write COG
    cog_fn = str(tmp_dir / "test_cog.tif")
    rioda_large.raster.to_raster(cog_fn, driver="COG", overviews="auto")
    # test COG zoom levels
    # return native resolution
    res = np.asarray(rioda_large.raster.res)
    da1 = artifact_data_catalog.get_rasterdataset(cog_fn, zoom_level=0)
    assert np.allclose(da1.raster.res, res)
    # reurn zoom level 1
    da1 = artifact_data_catalog.get_rasterdataset(
        cog_fn, zoom_level=(res[0] * 2, "degree")
    )
    assert np.allclose(da1.raster.res, res * 2)
    # test if file hase no overviews
    tif_fn = str(tmp_dir / "test_tif_no_overviews.tif")
    rioda_large.raster.to_raster(tif_fn, driver="GTiff")
    da1 = artifact_data_catalog.get_rasterdataset(tif_fn, zoom_level=(0.01, "degree"))
    xr.testing.assert_allclose(da1, rioda_large)
    # test if file has {variable} in path
    da1 = artifact_data_catalog.get_rasterdataset(
        "merit_hydro", zoom_level=(0.01, "degree")
    )
    assert isinstance(da1, xr.Dataset)


@pytest.mark.integration()
def test_rasterdataset_driver_kwargs(artifact_data_catalog: DataCatalog, tmp_dir: Path):
    era5 = artifact_data_catalog.get_rasterdataset("era5")
    path_zarr = tmp_dir / "era5.zarr"
    era5.to_zarr(path_zarr)
    data_dict = {
        "era5_zarr": {
            "data_type": "RasterDataset",
            "driver": {
                "name": "raster_xarray",
                "preprocess": "round_latlon",
            },
            "metadata": {
                "crs": 4326,
            },
            "uri": str(path_zarr),
        }
    }
    datacatalog = DataCatalog()
    datacatalog.from_dict(data_dict)
    era5_zarr = datacatalog.get_rasterdataset("era5_zarr")
    path_nc = tmp_dir / "era5.nc"
    era5.to_netcdf(path_nc)

    data_dict2 = {
        "era5_nc": {
            "data_type": "RasterDataset",
            "driver": {
                "name": "raster_xarray",
                "preprocess": "round_latlon",
            },
            "metadata": {
                "crs": 4326,
            },
            "uri": str(path_nc),
        }
    }
    datacatalog.from_dict(data_dict2)
    era5_nc = datacatalog.get_rasterdataset("era5_nc")
    assert era5_zarr.equals(era5_nc)
    cast(RasterDatasetSource, datacatalog.get_source("era5_zarr")).to_file(
        tmp_dir / "era5_copy.zarr"
    )


def test_rasterdataset_unit_attrs(data_catalog: DataCatalog):
    source = data_catalog.get_source("era5")
    attrs = {
        "temp": {"unit": "degrees C", "long_name": "temperature"},
        "temp_max": {"unit": "degrees C", "long_name": "maximum temperature"},
        "temp_min": {"unit": "degrees C", "long_name": "minimum temperature"},
    }
    source.metadata.attrs.update(**attrs)
    data_catalog.add_source("era5", source)
    raster = data_catalog.get_rasterdataset("era5")
    assert raster["temp"].attrs["unit"] == attrs["temp"]["unit"]
    assert raster["temp_max"].attrs["long_name"] == attrs["temp_max"]["long_name"]


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_geodataset(geoda, geodf, ts, tmpdir, data_catalog):
    fn_nc = str(tmpdir.join("test.nc"))
    fn_gdf = str(tmpdir.join("test.geojson"))
    fn_csv = str(tmpdir.join("test.csv"))
    fn_csv_locs = str(tmpdir.join("test_locs.xy"))
    geoda.vector.to_netcdf(fn_nc)
    geodf.to_file(fn_gdf, driver="GeoJSON")
    ts.to_csv(fn_csv)
    write_xy(fn_csv_locs, geodf)
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
    with pytest.raises(FileNotFoundError, match="No such file"):
        data_catalog.get_geodataset("no_file.geojson")
    da3 = data_catalog.get_geodataset(
        "test.nc",
        # only really care that the bbox doesn't intersect with anythign
        bbox=[12.5, 12.6, 12.7, 12.8],
        handle_nodata=NoDataStrategy.IGNORE,
    )
    assert da3 is None

    with pytest.raises(NoDataException):
        da3 = data_catalog.get_geodataset(
            "test.nc",
            # only really care that the bbox doesn't intersect with anythign
            bbox=[12.5, 12.6, 12.7, 12.8],
            handle_nodata=NoDataStrategy.RAISE,
        )

    with tempfile.TemporaryDirectory() as td:
        # Test nc file writing to file
        GeoDatasetAdapter(fn_nc).to_file(
            data_root=td, data_name="test", driver="netcdf"
        )
        GeoDatasetAdapter(fn_nc).to_file(
            data_root=tmpdir, data_name="test1", driver="netcdf", variables="test1"
        )
        GeoDatasetAdapter(fn_nc).to_file(data_root=td, data_name="test", driver="zarr")


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_geodataset_unit_attrs(data_catalog: DataCatalog):
    gtsm_dict = {"gtsmv3_eu_era5": data_catalog.get_source("gtsmv3_eu_era5").to_dict()}
    attrs = {
        "waterlevel": {
            "long_name": "sea surface height above mean sea level",
            "unit": "meters",
        }
    }
    gtsm_dict["gtsmv3_eu_era5"].update(dict(attrs=attrs))
    data_catalog.from_dict(gtsm_dict)
    gtsm_geodataarray = data_catalog.get_geodataset("gtsmv3_eu_era5")
    assert gtsm_geodataarray.attrs["long_name"] == attrs["waterlevel"]["long_name"]
    assert gtsm_geodataarray.attrs["unit"] == attrs["waterlevel"]["unit"]


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_geodataset_unit_conversion(data_catalog: DataCatalog):
    gtsm_geodataarray = data_catalog.get_geodataset("gtsmv3_eu_era5")
    gtsm_dict = {"gtsmv3_eu_era5": data_catalog.get_source("gtsmv3_eu_era5").to_dict()}
    gtsm_dict["gtsmv3_eu_era5"].update(dict(unit_mult=dict(waterlevel=1000)))
    datacatalog = DataCatalog()
    datacatalog.from_dict(gtsm_dict)
    gtsm_geodataarray1000 = datacatalog.get_geodataset("gtsmv3_eu_era5")
    assert gtsm_geodataarray1000.equals(gtsm_geodataarray * 1000)


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_geodataset_set_nodata(data_catalog: DataCatalog):
    gtsm_dict = {"gtsmv3_eu_era5": data_catalog.get_source("gtsmv3_eu_era5").to_dict()}
    gtsm_dict["gtsmv3_eu_era5"].update(dict(nodata=-99))
    datacatalog = DataCatalog()
    datacatalog.from_dict(gtsm_dict)
    ds = datacatalog.get_geodataset("gtsmv3_eu_era5")
    assert ds.vector.nodata == -99


def test_dataset_get_data(timeseries_ds, tmpdir):
    path = str(tmpdir.join("test.nc"))
    timeseries_ds.to_netcdf(path)
    dataset_adapter = DatasetAdapter(path=path, driver="netcdf")
    ds1 = dataset_adapter.get_data()
    assert isinstance(ds1, xr.Dataset)
    assert ds1.identical(timeseries_ds)


def test_dataset_to_file(timeseries_ds, tmpdir):
    path = str(tmpdir.join("test1.nc"))
    encoding = {k: {"zlib": True} for k in timeseries_ds.vector.vars}
    timeseries_ds.to_netcdf(path, encoding=encoding)
    dataset_adapter = DatasetAdapter(path=path, driver="netcdf")
    fn_out, driver = dataset_adapter.to_file(
        data_root=tmpdir, data_name="test2", driver="netcdf"
    )
    assert driver == "netcdf"
    assert fn_out == str(tmpdir.join("test2.nc"))
    ds2 = xr.open_dataset(fn_out)
    assert ds2.identical(timeseries_ds)

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
    assert ds_zarr.identical(timeseries_ds)

    with pytest.raises(ValueError, match="Dataset: Driver fake-driver unknown."):
        dataset_adapter.to_file(
            data_root=tmpdir, data_name="test4", driver="fake-driver"
        )


def test_dataset_read_data(tmpdir, timeseries_ds):
    zarr_path = str(tmpdir.join("zarr_data"))
    timeseries_ds.to_zarr(zarr_path)
    dataset_adapter = DatasetAdapter(path=zarr_path, driver="zarr")
    dataset_adapter.get_data(variables=["col1", "col2"])

    dataset_adapter = DatasetAdapter(path=zarr_path, driver="fake-driver")
    with pytest.raises(ValueError, match=r"Dataset: Driver fake-driver unknown"):
        dataset_adapter.get_data(variables=["col1", "col2"])


def test_dataset_set_nodata(tmpdir, timeseries_ds):
    path = str(tmpdir.join("test.nc"))
    timeseries_ds.to_netcdf(path)

    nodata = -999
    dataset_adapter = DatasetAdapter(path=path, driver="netcdf", nodata=nodata)
    ds = dataset_adapter.get_data()
    for k in ds.data_vars:
        assert ds[k].attrs["_FillValue"] == nodata

    nodata = {"col1": -999, "col2": np.nan}
    dataset_adapter = DatasetAdapter(path=path, driver="netcdf", nodata=nodata)
    ds = dataset_adapter.get_data()
    assert np.isnan(ds["col2"].attrs["_FillValue"])
    assert ds["col1"].attrs["_FillValue"] == nodata["col1"]


def test_dataset_apply_unit_conversion(tmpdir, timeseries_ds):
    path = str(tmpdir.join("test.nc"))
    timeseries_ds.to_netcdf(path)

    dataset_adapter = DatasetAdapter(
        path=path,
        unit_mult=dict(col1=1000),
    )
    ds1 = dataset_adapter.get_data()

    assert ds1["col1"].equals(timeseries_ds["col1"] * 1000)

    dataset_adapter = DatasetAdapter(path=path, unit_add={"time": 10})
    ds2 = dataset_adapter.get_data()
    assert ds2["time"][-1].values == np.datetime64("2020-12-31T00:00:10")


def test_dataset_set_metadata(tmpdir, timeseries_ds):
    path = str(tmpdir.join("test.nc"))
    timeseries_ds.to_netcdf(path)
    meta_data = {"col1": {"long_name": "column1"}, "col2": {"long_name": "column2"}}
    dataset_adapter = DatasetAdapter(path=path, meta=meta_data)
    ds = dataset_adapter.get_data()
    assert ds.attrs["col1"].get("long_name") == "column1"
    assert ds.attrs["col2"].get("long_name") == "column2"

    dataset_adapter = DatasetAdapter(path=path, attrs=meta_data)
    ds = dataset_adapter.get_data()
    assert ds["col1"].attrs["long_name"] == "column1"
    assert ds["col2"].attrs["long_name"] == "column2"

    da = timeseries_ds["col1"]
    da.name = "col1"

    path = str(tmpdir.join("da.nc"))
    da.to_netcdf(path)

    dataset_adapter = DatasetAdapter(
        path=path, attrs={"col1": {"long_name": "column1"}}
    )
    da = dataset_adapter.get_data()
    assert da.attrs["long_name"] == "column1"


def test_dataset_to_stac_catalog(tmpdir, timeseries_ds):
    path = str(tmpdir.join("test.nc"))
    timeseries_ds.to_netcdf(path)
    dataset_adapter = DatasetAdapter(path=path, name="timeseries_dataset")

    stac_catalog = dataset_adapter.to_stac_catalog()
    assert isinstance(stac_catalog, StacCatalog)
    stac_item = next(stac_catalog.get_items("timeseries_dataset"), None)
    assert list(stac_item.assets.keys())[0] == "test.nc"


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_geodataframe(geodf, tmpdir, data_catalog):
    fn_gdf = str(tmpdir.join("test.geojson"))
    fn_shp = str(tmpdir.join("test.shp"))
    geodf.to_file(fn_gdf, driver="GeoJSON")
    geodf.to_file(fn_shp)
    # test read geojson using total bounds
    gdf1 = data_catalog.get_geodataframe(fn_gdf, bbox=geodf.total_bounds)
    assert isinstance(gdf1, gpd.GeoDataFrame)
    assert np.all(gdf1 == geodf)
    # test read shapefile using total bounds
    gdf1 = data_catalog.get_geodataframe(fn_shp, bbox=geodf.total_bounds)
    assert isinstance(gdf1, gpd.GeoDataFrame)
    assert np.all(gdf1 == geodf)
    # testt read shapefile using mask
    mask = gpd.GeoDataFrame({"geometry": [box(*geodf.total_bounds)]}, crs=geodf.crs)
    gdf1 = hydromt.open_vector(fn_shp, geom=mask)
    assert np.all(gdf1 == geodf)
    # test read with buffer
    gdf1 = data_catalog.get_geodataframe(
        fn_gdf, bbox=geodf.total_bounds, buffer=1000, rename={"test": "test1"}
    )
    assert np.all(gdf1 == geodf)
    gdf1 = data_catalog.get_geodataframe(
        fn_shp, bbox=geodf.total_bounds, buffer=1000, rename={"test": "test1"}
    )
    assert np.all(gdf1 == geodf)

    # test nodata
    gdf1 = data_catalog.get_geodataframe(
        fn_gdf,
        # only really care that the bbox doesn't intersect with anythign
        bbox=[12.5, 12.6, 12.7, 12.8],
        predicate="within",
        handle_nodata=NoDataStrategy.IGNORE,
    )

    assert gdf1 is None

    with pytest.raises(NoDataException):
        gdf1 = data_catalog.get_geodataframe(
            fn_gdf,
            # only really care that the bbox doesn't intersect with anythign
            bbox=[12.5, 12.6, 12.7, 12.8],
            predicate="within",
            handle_nodata=NoDataStrategy.RAISE,
        )

    with pytest.raises(FileNotFoundError):
        data_catalog.get_geodataframe("no_file.geojson")


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_geodataframe_unit_attrs(data_catalog: DataCatalog):
    gadm_level1 = {"gadm_level1": data_catalog.get_source("gadm_level1").to_dict()}
    attrs = {"NAME_0": {"long_name": "Country names"}}
    gadm_level1["gadm_level1"].update(dict(attrs=attrs))
    data_catalog.from_dict(gadm_level1)
    gadm_level1_gdf = data_catalog.get_geodataframe("gadm_level1")
    assert gadm_level1_gdf["NAME_0"].attrs["long_name"] == "Country names"


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_dataframe(df, tmpdir, data_catalog):
    # Test reading csv
    fn_df = str(tmpdir.join("test.csv"))
    df.to_csv(fn_df)
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


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
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


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_dataframe_time(df_time, tmpdir, data_catalog):
    # Test time df
    fn_df_ts = str(tmpdir.join("test_ts.csv"))
    df_time.to_csv(fn_df_ts)
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


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
# TODO: Partially tested in "test_rasterio_driver", just missing the cache=True option.
# https://github.com/Deltares/hydromt/issues/897
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


@pytest.mark.skip(reason="Needs refactoring to Pydantic BaseModel.")
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


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_to_stac_geodataframe(geodf, tmpdir, data_catalog):
    fn_gdf = str(tmpdir.join("test.geojson"))
    geodf.to_file(fn_gdf, driver="GeoJSON")
    # geodataframe
    name = "gadm_level1"
    adapter = cast(GeoDataFrameAdapter, data_catalog.get_source(name))
    bbox, crs = adapter.get_bbox()
    gdf_stac_catalog = StacCatalog(id=name, description=name)
    gds_stac_item = StacItem(
        name,
        geometry=None,
        bbox=list(bbox),
        properties=adapter.meta,
        datetime=datetime(1, 1, 1),
    )
    gds_stac_asset = StacAsset(str(adapter.path))
    gds_base_name = basename(adapter.path)
    gds_stac_item.add_asset(gds_base_name, gds_stac_asset)

    gdf_stac_catalog.add_item(gds_stac_item)
    outcome = cast(
        StacCatalog, adapter.to_stac_catalog(on_error=ErrorHandleMethod.RAISE)
    )
    assert gdf_stac_catalog.to_dict() == outcome.to_dict()  # type: ignore
    adapter.crs = -3.14  # manually create an invalid adapter by deleting the crs
    assert adapter.to_stac_catalog(on_error=ErrorHandleMethod.SKIP) is None


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_to_stac_raster(data_catalog):
    # raster dataset
    name = "chirps_global"
    adapter = cast(RasterDatasetAdapter, data_catalog.get_source(name))
    bbox, crs = adapter.get_bbox()
    start_dt, end_dt = adapter.get_time_range(detect=True)
    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)
    raster_stac_catalog = StacCatalog(id=name, description=name)
    raster_stac_item = StacItem(
        name,
        geometry=None,
        bbox=list(bbox),
        properties=adapter.meta,
        datetime=None,
        start_datetime=start_dt,
        end_datetime=end_dt,
    )
    raster_stac_asset = StacAsset(str(adapter.path))
    raster_base_name = basename(adapter.path)
    raster_stac_item.add_asset(raster_base_name, raster_stac_asset)

    raster_stac_catalog.add_item(raster_stac_item)

    outcome = cast(
        StacCatalog, adapter.to_stac_catalog(on_error=ErrorHandleMethod.RAISE)
    )

    assert raster_stac_catalog.to_dict() == outcome.to_dict()  # type: ignore
    adapter.crs = -3.14  # manually create an invalid adapter by deleting the crs
    assert adapter.to_stac_catalog(on_error=ErrorHandleMethod.SKIP) is None


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_to_stac_geodataset(data_catalog):
    # geodataset
    name = "gtsmv3_eu_era5"
    adapter = cast(GeoDatasetAdapter, data_catalog.get_source(name))
    bbox, crs = adapter.get_bbox()
    start_dt, end_dt = adapter.get_time_range(detect=True)
    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)
    gds_stac_catalog = StacCatalog(id=name, description=name)
    gds_stac_item = StacItem(
        name,
        geometry=None,
        bbox=list(bbox),
        properties=adapter.meta,
        datetime=None,
        start_datetime=start_dt,
        end_datetime=end_dt,
    )
    gds_stac_asset = StacAsset(str(adapter.path))
    gds_base_name = basename(adapter.path)
    gds_stac_item.add_asset(gds_base_name, gds_stac_asset)

    gds_stac_catalog.add_item(gds_stac_item)

    outcome = cast(
        StacCatalog, adapter.to_stac_catalog(on_error=ErrorHandleMethod.RAISE)
    )
    assert gds_stac_catalog.to_dict() == outcome.to_dict()  # type: ignore
    adapter.crs = -3.14  # manually create an invalid adapter by deleting the crs
    assert adapter.to_stac_catalog(ErrorHandleMethod.SKIP) is None


@pytest.mark.skip(reason="Needs implementation of all raster Drivers.")
def test_to_stac_dataframe(df, tmpdir):
    fn_df = str(tmpdir.join("test.csv"))
    name = "test_dataframe"
    df.to_csv(fn_df)
    dc = DataCatalog().from_dict(
        {
            name: {
                "data_type": "DataFrame",
                "path": fn_df,
            }
        }
    )

    adapter = dc.get_source(name)

    with pytest.raises(
        NotImplementedError,
        match="DataframeAdapter does not support full stac conversion ",
    ):
        adapter.to_stac_catalog(on_error=ErrorHandleMethod.RAISE)

    assert adapter.to_stac_catalog(on_error=ErrorHandleMethod.SKIP) is None

    stac_catalog = StacCatalog(
        name,
        description=name,
    )
    stac_item = StacItem(
        name,
        geometry=None,
        bbox=[0, 0, 0, 0],
        properties=adapter.meta,
        datetime=datetime(1, 1, 1),
    )
    stac_asset = StacAsset(str(fn_df))
    stac_item.add_asset("hydromt_path", stac_asset)

    stac_catalog.add_item(stac_item)
    outcome = cast(
        StacCatalog, adapter.to_stac_catalog(on_error=ErrorHandleMethod.COERCE)
    )
    assert stac_catalog.to_dict() == outcome.to_dict()  # type: ignore
