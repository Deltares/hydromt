# -*- coding: utf-8 -*-
"""Tests for the vector submodule."""

import pytest
import numpy as np
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import Polygon, MultiPolygon

from hydromt.vector import GeoDataset

# from hydromt import vector


@pytest.fixture
def dummy_shp():
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


def test_ogr(tmpdir, dummy_shp):
    # Create a geodataset and an ogr compliant version of it
    ds = GeoDataset.from_gdf(dummy_shp)
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


def test_geo(geoda, geodf):
    # vector props
    assert geoda.vector.crs.to_epsg() == geodf.crs.to_epsg()
    assert np.dtype(geoda[geoda.vector.time_dim]).type == np.datetime64
    assert np.all(geoda.vector.geometry == geodf.geometry)
    # build from array
    da1 = GeoDataset.from_gdf(geodf, geoda)[geoda.name]
    assert np.all(da1 == geoda)
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
    # errors
    with pytest.raises(ValueError, match="gdf data type not understood"):
        GeoDataset.from_gdf(geoda)


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
