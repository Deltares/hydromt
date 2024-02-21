import os
from os.path import join
from pathlib import Path

import geopandas as gpd
import pytest

from hydromt import DataCatalog
from hydromt._typing.model_mode import ModelMode
from hydromt.models import MODELS, Region
from hydromt.models._region.specifyers import (
    BboxRegionSpecifyer,
    GeomFileRegionSpecifyer,
    GeomRegionSpecifyer,
)
from hydromt.models._region.specifyers.basin import (
    BasinIDSpecifyer,
    BasinIDsSpecifyer,
    BasinXYsSpecifyer,
)
from hydromt.models._region.specifyers.catalog import (
    GeomCatalogRegionSpecifyer,
    GridCatalogRegionSpecifyer,
)
from hydromt.models._region.specifyers.grid import (
    GridDataRegionSpecifyer,
    GridPathRegionSpecifyer,
)
from hydromt.models._region.specifyers.interbasin import (
    InterBasinGeomSpecifyer,
)
from hydromt.models._region.specifyers.model import ModelRegionSpecifyer
from hydromt.models._region.specifyers.subbasin import (
    SubBasinXYSpecifyer,
)


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
    r = Region({"geom": Path(geom_path)})
    # r.construct()
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


def test_region_from_model(tmpdir):
    model = MODELS.generic[0]
    root = str(tmpdir.join(model)) + "_test_region"
    if not os.path.isdir(root):
        os.mkdir(root)
    region_dict = {"model": root}
    region = Region(region_dict)
    assert isinstance(region._spec.spec, ModelRegionSpecifyer)


def test_region_from_catalog():
    region_dict = {"geom": "world"}
    region = Region(region_dict)
    assert isinstance(region._spec.spec, GeomCatalogRegionSpecifyer)


def test_region_from_grid_data(rioda, tmpdir):
    fn_grid = str(tmpdir.join("grid.tif"))
    rioda.raster.to_raster(fn_grid)
    region_dict = {"grid": rioda}
    region = Region(region_dict)
    assert isinstance(region._spec.spec, GridDataRegionSpecifyer)


def test_region_from_grid_file(rioda, tmpdir):
    fn_grid = Path(tmpdir.join("grid.tif"))
    rioda.raster.to_raster(fn_grid)
    region_dict = {"grid": fn_grid}
    region = Region(region_dict)
    assert isinstance(region._spec.spec, GridPathRegionSpecifyer)


def test_region_from_grid_catalog():
    region_dict = {"grid": "grid"}
    region = Region(region_dict)
    assert isinstance(region._spec.spec, GridCatalogRegionSpecifyer)


def test_region_from_basin_ids():
    region_dict = {"basin": [1001, 1002, 1003, 1004, 1005]}
    region = Region(region_dict)
    assert isinstance(region._spec.spec, BasinIDsSpecifyer)
    assert region._spec.spec.ids == [1001, 1002, 1003, 1004, 1005]  # type: ignore


def test_region_from_basin_id():
    region_dict = {"basin": 1001}
    region = Region(region_dict)
    assert isinstance(region._spec.spec, BasinIDSpecifyer)
    assert region._spec.spec.id == 1001  # type: ignore


def test_region_from_subbasin():
    region_dict = {
        "subbasin": [1.0, -1.0],
        "uparea": 5.0,
        "bounds": [0.0, -5.0, 3.0, 0.0],
    }
    region = Region(region_dict)
    assert isinstance(region._spec.spec, SubBasinXYSpecifyer)


def test_region_from_basin_xys():
    region_dict = {"basin": [[1.0, 1.5], [0.0, -1.0]]}
    region = Region(region_dict)
    assert isinstance(region._spec.spec, BasinXYsSpecifyer)


def test_region_from_interbasin(geodf):
    region_dict = {"interbasin": geodf}
    region = Region(region_dict)
    assert isinstance(region._spec.spec, InterBasinGeomSpecifyer)


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
