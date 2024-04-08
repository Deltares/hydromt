import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from hydromt import DataCatalog, raster
from hydromt.components.spatial import _parse_region, _parse_region_value
from hydromt.workflows.basin_mask import _check_size


def test_region_from_geom(world):
    _kind, region = _parse_region({"geom": world})
    assert isinstance(region["geom"], gpd.GeoDataFrame)


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_file(tmpdir, world):
    fn_gdf = str(tmpdir.join("world.geojson"))
    world.to_file(fn_gdf, driver="GeoJSON")
    _kind, region = _parse_region({"geom": fn_gdf})
    assert isinstance(region["geom"], gpd.GeoDataFrame)


@pytest.mark.skip("new driver implementation causes validation error")
def test_geom_from_cat(tmpdir, world):
    fn_gdf = str(tmpdir.join("world.geojson"))
    world.to_file(fn_gdf, driver="GeoJSON")
    cat = DataCatalog()
    cat.from_dict(
        {
            "world": {
                "path": fn_gdf,
                "data_type": "GeoDataFrame",
                "driver": "vector",
            },
        }
    )
    _kind, region = _parse_region({"geom": "world"}, data_catalog=cat)
    assert isinstance(region["geom"], gpd.GeoDataFrame)


def test_geom_from_points_fails(geodf):
    region = {"geom": geodf}
    with pytest.raises(ValueError, match=r"Region value.*"):
        _kind, region = _parse_region(region)


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_grid(rioda):
    _kind, region = _parse_region({"grid": rioda})
    assert isinstance(region["grid"], xr.DataArray)


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_grid_file(tmpdir, rioda):
    fn_grid = str(tmpdir.join("grid.tif"))
    rioda.raster.to_raster(fn_grid)
    _kind, region = _parse_region({"grid": fn_grid})
    assert isinstance(region["grid"], xr.DataArray)


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_grid_cat(tmpdir, rioda):
    fn_grid = str(tmpdir.join("grid.tif"))
    rioda.raster.to_raster(fn_grid)
    cat = DataCatalog()
    cat.from_dict(
        {
            "grid": {
                "path": fn_grid,
                "data_type": "RasterDataset",
                "driver": "raster",
            },
        }
    )
    _kind, region = _parse_region({"grid": "grid"}, data_catalog=cat)
    assert isinstance(region["grid"], xr.DataArray)


def test_region_from_basin_ids():
    region = {"basin": [1001, 1002, 1003, 1004, 1005]}
    kind, region = _parse_region(region)
    assert kind == "basin"
    assert region.get("basid") == [1001, 1002, 1003, 1004, 1005]


def test_region_from_basin_id():
    region = {"basin": 101}
    kind, region = _parse_region(region)
    assert kind == "basin"
    assert region.get("basid") == 101


def test_region_from_subbasin():
    region = {"subbasin": [1.0, -1.0], "uparea": 5.0, "bounds": [0.0, -5.0, 3.0, 0.0]}
    kind, region = _parse_region(region)
    assert kind == "subbasin"
    assert "xy" in region
    assert "bounds" in region


def test_region_from_basin_xy():
    region = {"basin": [[1.0, 1.5], [0.0, -1.0]]}
    _kind, region = _parse_region(region)
    assert "xy" in region


def test_region_from_inter_basin(geodf):
    region = {"interbasin": geodf}
    _kind, region = _parse_region(region)
    assert "xy" in region


# @pytest.mark.skip(reason="region from model not yet imported")
# def test_region_from_model(tmpdir, world, geodf, rioda):
#     # prepare test data
#     model = MODELS.generic[0]
#     root = str(tmpdir.join(model)) + "_test_region"
#     if not os.path.isdir(root):
#         os.mkdir(root)
##     region = {model: root}
#     kind, region = _parse_region(region)
#     assert kind == "model"


@pytest.mark.skip(reason="MODEL import does not work yet")
def test_region_from_unknown_errors():
    # model
    region = {"region": [0.0, -1.0]}
    with pytest.raises(ValueError, match=r"Region key .* not understood.*"):
        _parse_region(region)


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


def test_region_value_basin_ids():
    data_catalog = DataCatalog()
    array = np.array([1001, 1002, 1003, 1004, 1005])
    kwarg = _parse_region_value(array, data_catalog=data_catalog)
    assert kwarg.get("basid") == array.tolist()


def test_region_value_xy():
    data_catalog = DataCatalog()
    xy = (1.0, -1.0)
    kwarg = _parse_region_value(xy, data_catalog=data_catalog)
    assert kwarg.get("xy") == xy


def test_region_value_cat():
    data_catalog = DataCatalog()
    root = "./"
    kwarg = _parse_region_value(root, data_catalog=data_catalog)
    assert kwarg.get("root") == root
