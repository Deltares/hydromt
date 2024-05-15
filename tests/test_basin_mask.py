# -*- coding: utf-8 -*-
"""Tests for the hydromt.workflows.basin_mask."""

import logging
import os

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from hydromt import raster
from hydromt.models import MODELS
from hydromt.workflows.basin_mask import (
    _check_size,
    _parse_region_value,
    get_basin_geometry,
    parse_region,
)

logger = logging.getLogger("tets_basin")


def test_region(tmpdir, world, geodf, rioda, data_catalog):
    # prepare test data
    fn_gdf = str(tmpdir.join("world.geojson"))
    world.to_file(fn_gdf, driver="GeoJSON")
    fn_grid = str(tmpdir.join("grid.tif"))
    rioda.raster.to_raster(fn_grid)
    data_catalog.from_dict(
        {
            "world": {
                "path": fn_gdf,
                "data_type": "GeoDataFrame",
                "driver": "vector",
            },
            "grid": {
                "path": fn_grid,
                "data_type": "RasterDataset",
                "driver": "raster",
            },
        }
    )

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
    kind, region = parse_region({"geom": world})
    assert isinstance(region["geom"], gpd.GeoDataFrame)
    kind, region = parse_region({"geom": fn_gdf})
    assert isinstance(region["geom"], gpd.GeoDataFrame)
    kind, region = parse_region({"geom": "world"}, data_catalog=data_catalog)
    assert isinstance(region["geom"], gpd.GeoDataFrame)
    # geom:  points should fail
    region = {"geom": geodf}
    with pytest.raises(ValueError, match=r"Region value.*"):
        kind, region = parse_region(region)

    # grid
    kind, region = parse_region({"grid": rioda})
    assert isinstance(region["grid"], xr.DataArray)
    kind, region = parse_region({"grid": fn_grid})
    assert isinstance(region["grid"], xr.DataArray)
    kind, region = parse_region({"grid": "grid"}, data_catalog=data_catalog)
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


def test_region_value(data_catalog):
    array = np.array([1001, 1002, 1003, 1004, 1005])
    kwarg = _parse_region_value(array, data_catalog=data_catalog)
    assert kwarg.get("basid") == array.tolist()
    xy = (1.0, -1.0)
    kwarg = _parse_region_value(xy, data_catalog=data_catalog)
    assert kwarg.get("xy") == xy
    root = "./"
    kwarg = _parse_region_value(root, data_catalog=data_catalog)
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
    assert (
        "Loading very large spatial domain to derive a subbasin. "
        "Provide initial 'bounds' if this takes too long." in caplog.text
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_basin(data_catalog):
    ds = data_catalog.get_rasterdataset("merit_hydro_1k")
    gdf_bas_index = data_catalog.get_geodataframe("merit_hydro_index")
    bas_index = data_catalog.get_source("merit_hydro_index")

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
    assert np.isclose(gdf_bas.area.sum(), 0.16847222)

    gdf_bas, gdf_out = get_basin_geometry(
        ds, kind="subbasin", basin_index=bas_index, xy=[12.2051, 45.8331], strord=4
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.area.sum(), 0.001875)
    assert np.isclose(gdf_out.geometry.x, 12.17916667)
    assert np.isclose(gdf_out.geometry.y, 45.8041666)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="subbasin",
        basin_index=bas_index,
        xy=[[12.2051, 12.9788], [45.8331, 45.6973]],
        strord=5,
    )
    assert gdf_bas.index.size == 2
    assert np.isclose(gdf_bas.area.sum(), 0.021389)
    assert np.isclose(gdf_out.geometry.x[1], 12.970833333333266)
    assert np.isclose(gdf_out.geometry.y[1], 45.69583333333334)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="subbasin",
        xy=[12.2051, 45.8331],
        strord=4,
        bounds=gdf_bas.total_bounds,
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.area.sum(), 0.001875)
    assert np.isclose(gdf_out.geometry.x, 12.179167)
    assert np.isclose(gdf_out.geometry.y, 45.804167)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="basin",
        basin_index=gdf_bas_index,
        bbox=[12.6, 45.5, 12.9, 45.7],
        buffer=1,
    )
    assert gdf_bas.index.size == 30
    assert np.isclose(gdf_bas.area.sum(), 1.033125)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="basin",
        basin_index=gdf_bas_index,
        bbox=[12.6, 45.5, 12.9, 45.7],
        buffer=1,
        strord=4,
    )
    assert gdf_bas.index.size == 4
    assert np.isclose(gdf_bas.area.sum(), 1.03104167)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="subbasin",
        basin_index=gdf_bas_index,
        bbox=[12.2, 46.2, 12.4, 46.3],
        strord=6,
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.area.sum(), 0.198055)
    assert np.isclose(gdf_out.geometry.x, 12.295833)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="interbasin",
        basin_index=gdf_bas_index,
        bbox=[12.2, 46.2, 12.4, 46.3],
        strord=6,
    )
    assert gdf_bas.index.size == 1
    assert np.isclose(gdf_bas.area.sum(), 0.0172222)
    assert np.isclose(gdf_out.geometry.x, 12.295833)

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="interbasin",
        basin_index=gdf_bas_index,
        bbox=[12.8, 45.55, 12.9, 45.65],
        outlets=True,
    )
    assert gdf_bas.index.size == 13

    gdf_bas, gdf_out = get_basin_geometry(
        ds,
        kind="basin",
        basin_index=gdf_bas_index,
        bbox=[12.8, 45.55, 12.9, 45.65],
        outlets=True,
    )
    assert gdf_bas.index.size == 13

    msg = (
        'kind="outlets" has been deprecated, use outlets=True in combination with'
        + ' kind="basin" or kind="interbasin" instead.'
    )
    with pytest.warns(DeprecationWarning, match=msg):
        gdf_bas, gdf_out = get_basin_geometry(ds, kind="outlet")

    with pytest.raises(ValueError, match="Unknown kind: watershed,"):
        gdf_bas, gdf_out = get_basin_geometry(ds, kind="watershed")

    with pytest.raises(ValueError, match="Dataset variable stream_kwargs not in ds"):
        gdf_bas, gdf_out = get_basin_geometry(
            ds, kind="basin", stream_kwargs={"within": True}
        )
    with pytest.raises(
        ValueError, match='"kind=interbasin" requires either "bbox" or "geom"'
    ):
        gdf_bas, gdf_out = get_basin_geometry(
            ds,
            kind="interbasin",
        )
