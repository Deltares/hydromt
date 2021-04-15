# -*- coding: utf-8 -*-
"""Tests for the geo submodule."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from hydromt import geo


def test_geo(geoda, geodf):
    # geo props
    assert geoda.geo.crs.to_epsg() == geodf.crs.to_epsg()
    assert np.dtype(geoda[geoda.geo.time_dim]).type == np.datetime64
    assert np.all(geoda.geo.xcoords.values == geodf.geometry.x)
    assert np.all(geoda.geo.ycoords.values == geodf.geometry.y)
    # build from array
    da1 = geo.GeoDataset.from_gdf(geodf, geoda)[geoda.name]
    assert np.all(da1 == geoda)
    # to geopandas
    gdf1 = geoda.geo.to_gdf(reducer=np.mean)
    gdf2 = geoda.to_dataset().geo.to_gdf(reducer=np.mean)
    assert geoda.name in gdf1.columns
    assert np.all(gdf1.geometry.values == geodf.geometry.values)
    assert gdf1.crs == geodf.crs
    assert np.all(gdf1 == gdf2)
    # reproject
    da1 = geoda.geo.to_crs(3857)
    gdf1 = geodf.to_crs(3857)
    assert np.all(da1.geo.xcoords.values == gdf1.geometry.x.values)
    # errors
    with pytest.raises(ValueError, match="Unknown data type"):
        geo.GeoDataset.from_gdf(geoda)
    with pytest.raises(ValueError, match="only contain Point geometry"):
        geo.GeoDataset.from_gdf(geodf.buffer(1))
    with pytest.raises(ValueError, match="not found"):
        geo.GeoDataset.from_gdf(geodf, geoda.rename({"index": "missing"}))


def test_geo_clip(geoda, world):
    country = "Chile"
    geom = world[world["name"] == country]
    da1 = geoda.geo.clip_bbox(geom.total_bounds, buffer=0)
    assert np.all(da1["country"] == country)
    da1 = geoda.geo.clip_bbox(geom.total_bounds, create_sindex=True)
    assert np.all(da1["country"] == country)
    da1 = geoda.geo.clip_geom(geom.to_crs(3857))
    assert np.all(da1["country"] == country)
    da1 = geoda.geo.clip_bbox(geoda.geo.bounds)
    assert np.all(da1 == geoda)
