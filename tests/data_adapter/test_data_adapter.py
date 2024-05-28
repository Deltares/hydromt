# -*- coding: utf-8 -*-
"""Tests for the hydromt.data_adapter submodule."""

import glob
from datetime import datetime
from os.path import abspath, basename, dirname, join
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem

from hydromt import _compat as compat
from hydromt._typing import ErrorHandleMethod
from hydromt.data_adapter import (
    DatasetAdapter,
    GeoDataFrameAdapter,
    GeoDatasetAdapter,
    RasterDatasetAdapter,
)
from hydromt.data_catalog import DataCatalog
from hydromt.data_source import RasterDatasetSource
from hydromt.gis.utils import to_geographic_bbox

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


@pytest.mark.skip(
    "Needs implementation of https://github.com/Deltares/hydromt/issues/875"
)
@pytest.mark.integration()
def test_rasterdataset_zoomlevels(
    rioda_large: xr.DataArray, tmp_dir: Path, data_catalog: DataCatalog
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
    data_catalog.from_dict(yml_dict)
    rds: RasterDatasetSource = cast(RasterDatasetSource, data_catalog.get_source(name))
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
    cog_fn = str(tmp_dir / "test_cog.tif")
    rioda_large.raster.to_raster(cog_fn, driver="COG", overviews="auto")
    # test COG zoom levels
    # return native resolution
    res = np.asarray(rioda_large.raster.res)
    da1 = data_catalog.get_rasterdataset(cog_fn, zoom_level=0)
    assert np.allclose(da1.raster.res, res)
    # reurn zoom level 1
    da1 = data_catalog.get_rasterdataset(cog_fn, zoom_level=(res[0] * 2, "degree"))
    assert np.allclose(da1.raster.res, res * 2)
    # test if file hase no overviews
    tif_fn = str(tmp_dir / "test_tif_no_overviews.tif")
    rioda_large.raster.to_raster(tif_fn, driver="GTiff")
    da1 = data_catalog.get_rasterdataset(tif_fn, zoom_level=(0.01, "degree"))
    xr.testing.assert_allclose(da1, rioda_large)
    # test if file has {variable} in path
    da1 = data_catalog.get_rasterdataset("merit_hydro", zoom_level=(0.01, "degree"))
    assert isinstance(da1, xr.Dataset)


# TODO: migrate with https://github.com/Deltares/hydromt/issues/878
def test_dataset_get_data(timeseries_ds, tmpdir):
    path = str(tmpdir.join("test.nc"))
    timeseries_ds.to_netcdf(path)
    dataset_adapter = DatasetAdapter(path=path, driver="netcdf")
    ds1 = dataset_adapter.get_data()
    assert isinstance(ds1, xr.Dataset)
    assert ds1.identical(timeseries_ds)


# TODO: migrate with https://github.com/Deltares/hydromt/issues/878
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


# TODO: migrate with https://github.com/Deltares/hydromt/issues/878
def test_dataset_read_data(tmpdir, timeseries_ds):
    zarr_path = str(tmpdir.join("zarr_data"))
    timeseries_ds.to_zarr(zarr_path)
    dataset_adapter = DatasetAdapter(path=zarr_path, driver="zarr")
    dataset_adapter.get_data(variables=["col1", "col2"])

    dataset_adapter = DatasetAdapter(path=zarr_path, driver="fake-driver")
    with pytest.raises(ValueError, match=r"Dataset: Driver fake-driver unknown"):
        dataset_adapter.get_data(variables=["col1", "col2"])


# TODO: migrate with https://github.com/Deltares/hydromt/issues/878
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


# TODO: migrate with https://github.com/Deltares/hydromt/issues/878
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


# TODO: migrate with https://github.com/Deltares/hydromt/issues/878
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


# TODO: migrate with https://github.com/Deltares/hydromt/issues/878
def test_dataset_to_stac_catalog(tmpdir, timeseries_ds):
    path = str(tmpdir.join("test.nc"))
    timeseries_ds.to_netcdf(path)
    dataset_adapter = DatasetAdapter(path=path, name="timeseries_dataset")

    stac_catalog = dataset_adapter.to_stac_catalog()
    assert isinstance(stac_catalog, StacCatalog)
    stac_item = next(stac_catalog.get_items("timeseries_dataset"), None)
    assert list(stac_item.assets.keys())[0] == "test.nc"


@pytest.mark.skip(
    reason="Needs implementation https://github.com/Deltares/hydromt/issues/875."
)
def test_reads_slippy_map_output(tmp_dir: Path, rioda_large: xr.DataArray):
    # write vrt data
    name = "tiled"
    root = tmp_dir / name
    rioda_large.raster.to_xyz_tiles(
        root=root,
        tile_size=256,
        zoom_levels=[0],
    )
    cat = DataCatalog(str(root / f"{name}.yml"), cache=True)
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
