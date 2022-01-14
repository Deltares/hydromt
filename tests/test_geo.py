# -*- coding: utf-8 -*-
"""Tests for the vector submodule."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from hydromt import vector


def test_geo(geoda, geodf):
    # vector props
    assert geoda.vector.crs.to_epsg() == geodf.crs.to_epsg()
    assert np.dtype(geoda[geoda.vector.time_dim]).type == np.datetime64
    assert np.all(geoda.vector.xcoords.values == geodf.geometry.x)
    assert np.all(geoda.vector.ycoords.values == geodf.geometry.y)
    # build from array
    da1 = vector.GeoDataset.from_gdf(geodf, geoda)[geoda.name]
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
    assert np.all(da1.vector.xcoords.values == gdf1.geometry.x.values)
    # errors
    with pytest.raises(ValueError, match="Unknown data type"):
        vector.GeoDataset.from_gdf(geoda)
    with pytest.raises(ValueError, match="only contain Point geometry"):
        vector.GeoDataset.from_gdf(geodf.to_crs(3857).buffer(1))
    with pytest.raises(ValueError, match="not found"):
        vector.GeoDataset.from_gdf(geodf, geoda.rename({"index": "missing"}))


def test_geo_clip(geoda, world):
    country = "Chile"
    geom = world[world["name"] == country]
    da1 = geoda.vector.clip_bbox(geom.total_bounds, buffer=0)
    assert np.all(da1["country"] == country)
    da1 = geoda.vector.clip_bbox(geom.total_bounds, create_sindex=True)
    assert np.all(da1["country"] == country)
    da1 = geoda.vector.clip_geom(geom.to_crs(3857))
    assert np.all(da1["country"] == country)
    da1 = geoda.vector.clip_bbox(geoda.vector.bounds)
    assert np.all(da1 == geoda)
