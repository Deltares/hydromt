import os

import geopandas as gpd
import pytest
import xarray as xr

from hydromt import DataCatalog
from hydromt.models import MODELS
from hydromt.models.api import parse_region
from hydromt.region.region import Region
from hydromt.region.specifyers import BboxRegionSpecifyer


def test_bbox_region():
    region = {"bbox": [0.0, -5.0, 3.0, 0.0]}
    spec = Region._parse_region(region).spec
    assert isinstance(spec, BboxRegionSpecifyer), type(spec)


def test_region_unknown_key_errors():
    region = {"region": [0.0, -1.0]}
    with pytest.raises(ValueError, match=r"Unknown region kind.*"):
        _ = Region(region)


@pytest.mark.skip(reason="model region spec not yet implemented")
def test_region_from_model(tmpdir):
    model = MODELS.generic[0]
    root = str(tmpdir.join(model)) + "_test_region"
    if not os.path.isdir(root):
        os.mkdir(root)
    region = {model: root}
    kind, region = parse_region(region)
    assert kind == "model"


@pytest.mark.skip(reason="Needs implementation of subbasin region.")
def test_region_from_catalog(test_cat):
    kind, region = parse_region({"geom": "world"}, data_catalog=test_cat)
    assert isinstance(region["geom"], gpd.GeoDataFrame)


@pytest.mark.skip(reason="Needs RasterDataset implementation")
def test_region_from_geom_file(tmpdir, world):
    fn_gdf = str(tmpdir.join("world.geojson"))
    world.to_file(fn_gdf, driver="GeoJSON")
    kind, region = parse_region({"geom": fn_gdf})
    assert isinstance(region["geom"], gpd.GeoDataFrame)


def test_region_from_geom(world):
    kind, region = parse_region({"geom": world})
    assert isinstance(region["geom"], gpd.GeoDataFrame)


def test_region_from_geom_points_fails(geodf):
    region = {"geom": geodf}
    with pytest.raises(ValueError, match=r"Region value.*"):
        kind, region = parse_region(region)


@pytest.mark.skip(reason="Needs RasterDataset implementation")
def test_region_from_grid_data(rioda, tmpdir):
    fn_grid = str(tmpdir.join("grid.tif"))
    rioda.raster.to_raster(fn_grid)
    kind, region = parse_region({"grid": rioda})
    assert isinstance(region["grid"], xr.DataArray)


@pytest.mark.skip(reason="Needs RasterDataset implementation")
def test_region_from_grid_file(rioda, tmpdir):
    fn_grid = str(tmpdir.join("grid.tif"))
    rioda.raster.to_raster(fn_grid)
    kind, region = parse_region({"grid": fn_grid})
    assert isinstance(region["grid"], xr.DataArray)


@pytest.mark.skip(reason="Needs RasterDataset implementation")
def test_region_from_grid_catalog(test_cat):
    kind, region = parse_region({"grid": "grid"}, data_catalog=test_cat)
    assert isinstance(region["grid"], xr.DataArray)


def test_region_from_basin_ids():
    region = {"basin": [1001, 1002, 1003, 1004, 1005]}
    kind, region = parse_region(region)
    assert kind == "basin"
    assert region.get("basid") == [1001, 1002, 1003, 1004, 1005]


def test_region_from_basin_id():
    region = {"basin": 101}
    kind, region = parse_region(region)
    assert kind == "basin"
    assert region.get("basid") == 101


def test_region_from_subbasin(geodf):
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


@pytest.fixture()
def test_cat(tmpdir, world, geodf, rioda):
    # prepare test data
    fn_gdf = str(tmpdir.join("world.geojson"))
    world.to_file(fn_gdf, driver="GeoJSON")
    fn_grid = str(tmpdir.join("grid.tif"))
    rioda.raster.to_raster(fn_grid)
    cat = DataCatalog()
    cat.from_dict(
        {
            "world": {
                "path": fn_gdf,
                "data_type": "GeoDataFrame",
                "driver": "vector",
            },
            # "grid": {
            #     "path": fn_grid,
            #     "data_type": "RasterDataset",
            #     "driver": "raster",
            # },
        }
    )
    return cat
