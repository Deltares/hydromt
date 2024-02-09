import os

import geopandas as gpd
import pytest
import xarray as xr

from hydromt import DataCatalog
from hydromt._validators.region import BboxRegionSpecifyer
from hydromt.models import MODELS
from hydromt.models._v1.model_region import ModelRegion
from hydromt.models.api import parse_region


def test_bbox_region():
    region = {"outlet": [0.0, -5.0, 3.0, 0.0]}
    spec = ModelRegion._parse_region(region)
    assert isinstance(spec, BboxRegionSpecifyer)


def test_region(tmpdir, world, geodf, rioda):
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
        _ = ModelRegion(region, data_catalog=None)

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
    kind, region = parse_region({"geom": "world"}, data_catalog=cat)
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
    kind, region = parse_region({"grid": "grid"}, data_catalog=cat)
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
