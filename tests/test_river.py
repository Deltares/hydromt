"""Test hydromt.river submodule"""
from _pytest.monkeypatch import V
import pytest
import numpy as np
import xarray as xr
from hydromt import flw
import geopandas as gpd

import hydromt
import pyflwdir


def test_river_width():
    data_catalog = hydromt.DataCatalog()
    ds = data_catalog.get_rasterdataset("merit_hydro", bbox=(12.502213,45.6385,12.551652,45.673632))
    flw = hydromt.flw.flwdir_from_da(ds['flwdir'], ftype="d8")
    feats = flw.streams(min_sto=4)
    gdf_stream = gpd.GeoDataFrame.from_features(feats)
    #gdf_stream.to_file('streams.geojson', driver='GeoJSON')
    da_rivmask = ds['rivwth'] != 0

    hydromt.workflows.river_width(gdf_stream,da_rivmask)
    



