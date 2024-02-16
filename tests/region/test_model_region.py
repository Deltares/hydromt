import os
from os.path import join

import geopandas as gpd
import pytest
import xarray as xr

from hydromt import DataCatalog
from hydromt._typing.model_mode import ModelMode
from hydromt.models import MODELS
from hydromt.models.api import parse_region
from hydromt.region._specifyers import (
    BboxRegionSpecifyer,
    GeomFileRegionSpecifyer,
    GeomRegionSpecifyer,
)
from hydromt.region.region import Region


def test_write_region_read_mode(tmpdir):
    path = join(tmpdir, "region.geojson")
    region = {"bbox": [0.0, -5.0, 3.0, 0.0]}
    r = Region(region)
    r.construct()
    with pytest.raises(
        ValueError, match="Cannot write region when not in writing mode"
    ):
        r.write(path, ModelMode.READ)


def test_write_region_forced_write_mode(tmpdir):
    path = join(tmpdir, "region.geojson")
    region1 = {"bbox": [0.0, -5.0, 3.0, 0.0]}
    region2 = {"bbox": [0.0, 0.0, 24.0, 42.0]}
    r1 = Region(region1)
    r1.construct()
    r2 = Region(region2)
    r2.construct()
    r1.write(path, ModelMode.WRITE)
    r1_geom = gpd.GeoDataFrame.from_file(path)
    r2.write(path, ModelMode.FORCED_WRITE)
    r2_geom = gpd.GeoDataFrame.from_file(path)
    assert all(r1_geom.ne(r2_geom))


def test_write_region_append_mode(tmpdir):
    path = join(tmpdir, "region.geojson")
    region = {"bbox": [0.0, -5.0, 3.0, 0.0]}
    r = Region(region)
    r.construct()
    r.write(path, ModelMode.WRITE)
    with pytest.raises(ValueError, match="mode was not in forced override mode"):
        r.write(path, ModelMode.APPEND)


def test_bbox_region():
    region = {"bbox": [0.0, -5.0, 3.0, 0.0]}
    r = Region(region)
    r.construct()
    assert isinstance(r._spec.spec, BboxRegionSpecifyer)


def test_invalid_bbox_region():
    region = {"bbox": [0.0, -5.0, -1.0, -10.0]}
    with pytest.raises(
        ValueError, match=r".*Value error, xmin.* should be strictly less than.*"
    ):
        _ = Region(region)


def test_region_from_geom_file(tmpdir, world):
    geom_path = str(tmpdir.join("world.geojson"))
    world.to_file(geom_path, driver="GeoJSON")
    r = Region({"geom": geom_path})
    r.construct()
    assert isinstance(r._spec.spec, GeomFileRegionSpecifyer)


def test_region_unknown_key_errors():
    region = {"region": [0.0, -1.0]}
    with pytest.raises(ValueError, match=r"Unknown region kind.*"):
        _ = Region(region)


def test_region_from_geom(world):
    r = Region({"geom": world})
    assert isinstance(r._spec.spec, GeomRegionSpecifyer)


def test_region_from_geom_points_fails(geodf):
    with pytest.raises(ValueError, match=r".*validation error for RegionSpecifyer.*"):
        _ = Region({"geom": geodf})


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


@pytest.mark.skip(reason="Needs implementation of subbasin region.")
def test_region_from_basin_ids():
    region = {"basin": [1001, 1002, 1003, 1004, 1005]}
    kind, region = parse_region(region)
    assert kind == "basin"
    assert region.get("basid") == [1001, 1002, 1003, 1004, 1005]


@pytest.mark.skip(reason="Needs implementation of subbasin region.")
def test_region_from_basin_id():
    region = {"basin": 101}
    kind, region = parse_region(region)
    assert kind == "basin"
    assert region.get("basid") == 101


@pytest.mark.skip(reason="Needs implementation of subbasin region.")
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
