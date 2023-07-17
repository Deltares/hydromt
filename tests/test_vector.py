# -*- coding: utf-8 -*-
"""Tests for the vector submodule."""

import numpy as np
import pytest
import xarray as xr
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon

from hydromt.vector import GeoDataArray, GeoDataset


@pytest.fixture()
def gdf():
    geom = [
        Polygon(((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))),
        MultiPolygon(
            [
                Polygon(((1, 0), (2, 0), (2, 1), (1, 1), (1, 0))),
                Polygon(((2, 0), (3, 0), (3, 1), (2, 1), (2, 0))),
            ]
        ),
    ]
    attrs = [
        {"Roman": "I"},
        {"Roman": "II"},
    ]
    gdf = GeoDataFrame(data=attrs, geometry=geom, crs=CRS.from_epsg(4326))
    return gdf


def test_nodata(geoda):
    assert np.isnan(geoda.vector.nodata)
    # set with integer -> should be converted to geoda dtype
    geoda.vector.set_nodata(np.int32(-9999))
    assert geoda.vector.nodata == -9999
    assert type(geoda.vector.nodata) == geoda.dtype
    # remove nodata
    geoda.vector.set_nodata(None)
    assert geoda.vector.nodata is None


def test_ogr(tmpdir, gdf):
    # Create a geodataset and an ogr compliant version of it
    ds = GeoDataset.from_gdf(gdf)
    oc = ds.vector.ogr_compliant()

    # Assert some ogr compliant stuff
    assert oc.ogr_layer_type.upper() == "MULTIPOLYGON"
    assert list(oc.dims)[0] == "index"
    assert len(oc.Roman) == 2

    # Write and load
    fn = str(tmpdir.join("dummy_ogr.nc"))
    ds.vector.to_netcdf(fn, ogr_compliant=True)
    ds1 = GeoDataset.from_netcdf(fn)
    assert np.all(ds.vector.geometry == ds1.vector.geometry)


def test_vector(geoda, geodf):
    # vector props
    assert geoda.vector.crs.to_epsg() == geodf.crs.to_epsg()
    assert np.dtype(geoda[geoda.vector.time_dim]).type == np.datetime64
    assert np.all(geoda.vector.geometry == geodf.geometry)
    # to geopandas
    gdf1 = geoda.vector.to_gdf(reducer=np.mean)
    gdf2 = geoda.to_dataset().vector.to_gdf(reducer=np.mean)
    assert geoda.name in gdf1.columns
    assert np.all(gdf1.geometry.values == geodf.geometry.values)
    assert gdf1.crs == geodf.crs
    assert np.all(gdf1 == gdf2)
    # reproject
    da1 = geoda.vector.to_crs(3857)
    gdf1 = geodf.to_crs(3857)
    assert np.all(da1.vector.geometry == gdf1.geometry)


def test_from_gdf(geoda, geodf):
    geoda0 = geoda.reset_coords(drop=True)  # drop geometries
    dims = list(geoda0.dims)
    coords = geoda0.coords
    # build from GeoDataFrame
    da1 = GeoDataArray.from_gdf(geodf, geoda0)
    assert np.all(geodf.geometry == da1.vector.geometry)
    assert all([c in da1.coords for c in geodf.columns])
    ds1 = GeoDataset.from_gdf(geodf, geoda0)  # idem for dataset
    xr.testing.assert_equal(ds1[geoda.name], da1)
    # build from GeoSeries
    da1 = GeoDataArray.from_gdf(geodf["geometry"], geoda0)
    assert np.all(geodf.geometry == da1.vector.geometry)
    ds1 = GeoDataset.from_gdf(geodf["geometry"], geoda0)  # idem for dataset
    xr.testing.assert_equal(ds1[geoda.name], da1)
    # test with numpy array
    ds1 = GeoDataset.from_gdf(geodf, {"test": (dims, geoda0.values)}, coords=coords)
    assert isinstance(ds1, xr.Dataset)
    assert np.all(geodf.geometry == ds1.vector.geometry)
    da1 = GeoDataArray.from_gdf(geodf, geoda0.values, coords=coords)  # idem for array
    xr.testing.assert_equal(ds1["test"], da1)
    # test with different merging strategies
    da1 = GeoDataArray.from_gdf(geodf, geoda0.sel(index=[0, 1, 2]))
    assert np.all(np.isnan(da1.sel(index=[3, 4])))
    assert np.all(geodf.geometry == da1.vector.geometry)
    assert np.all(da1.index == geodf.index)
    ds1 = GeoDataset.from_gdf(geodf, geoda0.sel(index=[0, 1, 2]))  # idem for dataset
    xr.testing.assert_equal(ds1[geoda.name], da1)
    # inner join
    da1 = GeoDataArray.from_gdf(
        geodf.loc[[2, 3, 4]], geoda0.sel(index=[0, 1, 2]), merge_index="inner"
    )
    assert np.all(np.isin(da1.index, [2]))
    ds1 = GeoDataset.from_gdf(
        geodf.loc[[2, 3, 4]], geoda0.sel(index=[0, 1, 2]), merge_index="inner"
    )  # idem for dataset
    xr.testing.assert_equal(ds1[geoda.name], da1)
    # test without GeoDataset from gdf only
    ds1 = GeoDataset.from_gdf(geodf)
    assert isinstance(ds1, xr.Dataset)
    assert np.all(geodf.geometry == ds1.vector.geometry)
    # errors GeoDataset
    with pytest.raises(ValueError, match="gdf data type not understood"):
        GeoDataset.from_gdf(geoda)
    with pytest.raises(TypeError, match="data_vars should be a dict-like"):
        GeoDataset.from_gdf(geodf, data_vars=1)
    with pytest.raises(ValueError, match="Index dimension city not found in data_vars"):
        GeoDataset.from_gdf(geodf, geoda0, index_dim="city")
    with pytest.raises(ValueError, match="x is not a valid value for 'merge_index'"):
        GeoDataset.from_gdf(geodf, geoda0, merge_index="x")
    with pytest.raises(ValueError, match="No common indices found between gdf"):
        ds1 = GeoDataset.from_gdf(
            geodf.loc[[3]], geoda0.sel(index=[0]), merge_index="inner"
        )
    # errors GeoDataArray
    with pytest.raises(ValueError, match="gdf data type not understood"):
        GeoDataArray.from_gdf(geoda, geoda)
    with pytest.raises(ValueError, match="Index dimension city not found in data_vars"):
        GeoDataset.from_gdf(geodf, geoda0, index_dim="city")
    with pytest.raises(ValueError, match="x is not a valid value for 'merge_index'"):
        GeoDataArray.from_gdf(geodf, geoda0, merge_index="x")
    with pytest.raises(ValueError, match="No common indices found between gdf"):
        ds1 = GeoDataArray.from_gdf(
            geodf.loc[[3]], geoda0.sel(index=[0]), merge_index="inner"
        )


def test_geo_clip(geoda, world):
    country = "Chile"
    geom = world[world["name"] == country]
    da1 = geoda.vector.clip_bbox(geom.total_bounds, buffer=0)
    assert np.all(da1["country"] == country)
    da1 = geoda.vector.clip_bbox(geom.total_bounds)
    assert np.all(da1["country"] == country)
    da1 = geoda.vector.clip_geom(geom.to_crs(3857))
    assert np.all(da1["country"] == country)
    da1 = geoda.vector.clip_bbox(geoda.vector.bounds)
    assert np.all(da1 == geoda)
