# -*- coding: utf-8 -*-
"""Tests for the hydromt.workflows.basin_mask"""

import logging
import os
import warnings

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

import hydromt
from hydromt import raster
from hydromt.models import MODELS
from hydromt.workflows.basin_mask import (
    _check_size,
    _parse_region_value,
    get_basin_geometry,
    parse_region,
)

logger = logging.getLogger("tets_basin")


def test_region(tmpdir, world, geodf, rioda):
    # model
    region = {"region": [0.0, -1.0]}
    with pytest.raises(ValueError, match=r"Region key .* not understood.*"):
        parse_region(region)

    model = MODELS.generic[0]
    root = str(tmpdir.join(model)) + "_test_region"
    if not os.path.isdir(root):
        os.mkdir(root)
    region = {model: root}
    kind, region = parse_region(region)
    assert kind == "model"

    # geom
    region = {"geom": world}
    kind, region = parse_region(region)
    assert kind == "geom"
    assert isinstance(region["geom"], gpd.GeoDataFrame)
    fn_gdf = str(tmpdir.join("world.geojson"))
    world.to_file(fn_gdf, driver="GeoJSON")
    region = {"geom": fn_gdf}
    kind, region = parse_region(region)
    assert kind == "geom"
    assert isinstance(region["geom"], gpd.GeoDataFrame)
    # geom:  points should fail
    region = {"geom": geodf}
    with pytest.raises(ValueError, match=r"Region value.*"):
        kind, region = parse_region(region)

    # grid
    region = {"grid": rioda}
    kind, region = parse_region(region)
    assert kind == "grid"
    assert isinstance(region["grid"], xr.DataArray)
    fn_grid = str(tmpdir.join("grid.tif"))
    rioda.raster.to_raster(fn_grid)
    region = {"grid": fn_grid}
    kind, region = parse_region(region)
    assert isinstance(region["grid"], xr.DataArray)

    # basid
    region = {"basin": [1001, 1002, 1003, 1004, 1005]}
    kind, region = parse_region(region)
    assert kind == "basin"
    assert region.get("basid") == [1001, 1002, 1003, 1004, 1005]
    region = {"basin": 101}
    kind, region = parse_region(region)
    assert kind == "basin"
    assert region.get("basid") == 101

    # bbox
    region = {"outlet": [0.0, -5.0, 3.0, 0.0]}
    kind, region = parse_region(region)
    assert kind == "outlet"
    assert "bbox" in region

    # xy
    region = {"subbasin": [1.0, -1.0], "uparea": 5.0, "bounds": [0.0, -5.0, 3.0, 0.0]}
    kind, region = parse_region(region)
    assert kind == "subbasin"
    assert "xy" in region
    assert "bounds" in region
    region = {"basin": [[1.0, 1.5], [0.0, -1.0]]}
    kind, region = parse_region(region)
    assert "xy" in region
    region = {"interbasin": geodf}
    kind, region = parse_region(region)
    assert "xy" in region


def test_region_value():
    array = np.array([1001, 1002, 1003, 1004, 1005])
    kwarg = _parse_region_value(array)
    assert kwarg.get("basid") == array.tolist()
    xy = (1.0, -1.0)
    kwarg = _parse_region_value(xy)
    assert kwarg.get("xy") == xy
    root = "./"
    kwarg = _parse_region_value(root)
    assert kwarg.get("root") == root


def test_check_size(caplog):
    test_raster = raster.full_from_transform(
        transform=[0.5, 0.0, 3.0, 0.0, -0.5, -9.0],
        shape=(13000, 13000),
        nodata=-1,
        name="test",
        crs=4326,
        lazy=True,  # create lazy dask array instead of numpy array
    )
    _check_size(test_raster)
    assert "Loading very large spatial domain to derive a subbasin. "
    "Provide initial 'bounds' if this takes too long." in caplog.text


def test_basin(caplog):
    data_catalog = hydromt.DataCatalog(logger=logger)
    ds = data_catalog.get_rasterdataset("merit_hydro")
    gdf_bas_index = data_catalog.get_geodataframe("merit_hydro_index")
    bas_index = data_catalog["merit_hydro_index"]

    with pytest.raises(ValueError, match=r"No basins found"):
        gdf_bas, gdf_out = get_basin_geometry(
            ds,
            kind="basin",
            basid=0,  # basin ID should be > 0
        )

    gdf_bas, gdf_out = get_basin_geometry(
        ds.drop_vars("basins"),
        kind="basin",
        xy=[12.2051, 45.8331],
        buffer=1,
    )
    assert gdf_out is None
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.to_crs(3857).area.sum(), 9346337868.28675)

    gdf_bas, gdf_out = get_basin_geometry(
        ds, kind="subbasin", basin_index=bas_index, xy=[12.2051, 45.8331], strord=4
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.to_crs(3857).area.sum(), 8.277817e09)
    assert np.isclose(gdf_out.geometry.x, 12.205417)
    assert np.isclose(gdf_out.geometry.y, 45.83375)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="subbasin",
        basin_index=bas_index,
        xy=[[12.2051, 12.9788], [45.8331, 45.6973]],
        strord=5,
    )
    assert gdf_bas.index.size == 2
    assert np.isclose(gdf_bas.to_crs(3857).area.sum(), 8.446160e09)
    assert np.isclose(gdf_out.geometry.x[1], 12.97292)
    assert np.isclose(gdf_out.geometry.y[1], 45.69958)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="subbasin",
        xy=[12.2051, 45.8331],
        strord=4,
        bounds=gdf_bas.total_bounds,
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.to_crs(3857).area.sum(), 8.277817e09)
    assert np.isclose(gdf_out.geometry.x, 12.205417)
    assert np.isclose(gdf_out.geometry.y, 45.83375)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="basin",
        basin_index=gdf_bas_index,
        bbox=[12.6, 45.5, 12.9, 45.7],
        buffer=1,
    )
    assert gdf_bas.index.size == 470
    assert np.isclose(gdf_bas.to_crs(3857).area.sum(), 18433536552.16195)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="basin",
        basin_index=gdf_bas_index,
        bbox=[12.6, 45.5, 12.9, 45.7],
        buffer=1,
        strord=4,
    )
    assert gdf_bas.index.size == 6
    assert np.isclose(gdf_bas.to_crs(3857).area.sum(), 18407888488.828384)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="subbasin",
        basin_index=gdf_bas_index,
        bbox=[12.2, 46.2, 12.4, 46.3],
        strord=8,
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.to_crs(3857).area.sum(), 3569393882.735242)
    assert np.isclose(gdf_out.geometry.x, 12.300417)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="interbasin",
        basin_index=gdf_bas_index,
        bbox=[12.2, 46.2, 12.4, 46.3],
        strord=8,
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.to_crs(3857).area.sum(), 307314959.5972775)
    assert np.isclose(gdf_out.geometry.x, 12.300417)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="interbasin",
        basin_index=gdf_bas_index,
        bbox=[12.8, 45.55, 12.9, 45.65],
        outlets=True,
    )
    assert gdf_bas.index.size == 180

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="basin",
        basin_index=gdf_bas_index,
        bbox=[12.8, 45.55, 12.9, 45.65],
        outlets=True,
    )
    assert gdf_bas.index.size == 180

    msg = 'kind="outlets" has been deprecated, use outlets=True in combination with kind="basin" or kind="interbasin" instead.'
    with pytest.warns(DeprecationWarning, match=msg) as record:
        gdf_bas, gdf_out = get_basin_geometry(ds, kind="outlet")

    with pytest.raises(ValueError):
        gdf_bas, gdf_out = get_basin_geometry(ds, kind="watershed")

    with pytest.raises(ValueError):
        gdf_bas, gdf_out = get_basin_geometry(
            ds, kind="basin", stream_kwargs={"within": True}
        )
    with pytest.raises(ValueError):
        gdf_bas, gdf_out = get_basin_geometry(
            ds,
            kind="interbasin",
        )
