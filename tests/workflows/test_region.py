import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import xugrid as xu
from pytest_mock import MockerFixture

from hydromt import DataCatalog
from hydromt.model.model import Model
from hydromt.model.processes.region import (
    _parse_region_value,
    parse_region_basin,
    parse_region_geom,
    parse_region_grid,
    parse_region_mesh,
    parse_region_other_model,
)


def test_region_from_geom(world):
    region = parse_region_geom({"geom": world}, crs=None)
    assert world is region


@pytest.mark.skip("Needs new driver implementation")
def test_region_from_file(tmpdir, world):
    fn_gdf = str(tmpdir.join("world.geojson"))
    world.to_file(fn_gdf, driver="GeoJSON")
    region = parse_region_geom({"geom": fn_gdf}, crs=None, data_catalog=DataCatalog())
    gpd.testing.assert_geodataframe_equal(world, region)


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
    region = parse_region_geom({"geom": "world"}, crs=None, data_catalog=cat)
    assert isinstance(region["geom"], gpd.GeoDataFrame)


def test_geom_from_points_fails(geodf):
    region = {"geom": geodf}
    with pytest.raises(ValueError, match=r"Region value.*"):
        region = parse_region_geom(region=region, crs=None)


def test_region_from_grid(rioda):
    region = parse_region_grid({"grid": rioda}, data_catalog=None)
    xr.testing.assert_equal(region, rioda)


@pytest.mark.skip("new driver implementation causes validation error")
def test_region_from_grid_file(tmpdir, rioda):
    fn_grid = str(tmpdir.join("grid.tif"))
    rioda.raster.to_raster(fn_grid)
    region = parse_region_grid({"grid": fn_grid}, data_catalog=DataCatalog())
    xr.testing.assert_equal(region, rioda)


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
    region = parse_region_grid({"grid": "grid"}, data_catalog=cat)
    xr.testing.assert_equal(region, rioda)


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_basin_ids():
    region = {"basin": [1001, 1002, 1003, 1004, 1005]}
    region = parse_region_basin(
        region,
        data_catalog=DataCatalog(),
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
    )
    assert region.get("basid") == [1001, 1002, 1003, 1004, 1005]


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_basin_id():
    region = {"basin": 101}
    region = parse_region_basin(
        region,
        data_catalog=DataCatalog(),
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
    )
    assert region.get("basid") == 101


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_subbasin():
    region = {"subbasin": [1.0, -1.0], "uparea": 5.0, "bounds": [0.0, -5.0, 3.0, 0.0]}
    region = parse_region_basin(
        region,
        data_catalog=DataCatalog(),
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
    )
    assert "xy" in region
    assert "bounds" in region


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_basin_xy():
    region = {"basin": [[1.0, 1.5], [0.0, -1.0]]}
    region = parse_region_basin(
        region,
        data_catalog=DataCatalog(),
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
    )
    assert "xy" in region


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_inter_basin(geodf):
    region = {"interbasin": geodf}
    region = parse_region_basin(
        region,
        data_catalog=DataCatalog(),
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
    )
    assert "xy" in region


def test_region_from_model(tmpdir, world, mocker: MockerFixture):
    model = mocker.Mock(spec=Model, region=world)
    plugins = mocker.patch("hydromt.model.processes.region.PLUGINS")
    plugins.model_plugins = {"Model": mocker.Mock(return_value=model)}
    region = {Model.__name__: tmpdir}
    read_model = parse_region_other_model(region=region)
    assert read_model is model
    assert read_model.region is world


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


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_mesh(griduda):
    mesh = parse_region_mesh({"mesh": griduda})
    assert mesh == griduda


def test_parse_region_mesh_path(mocker: MockerFixture):
    ugrid_mock = mocker.Mock(spec_set=xu.UgridDataset)
    xu_open_dataset_mock = mocker.patch(
        "hydromt.model.processes.region.xu.open_dataset", return_value=ugrid_mock
    )
    mocker.patch("hydromt.model.processes.region.isfile", return_value=True)
    mesh = parse_region_mesh({"mesh": "path/to/mesh.nc"})
    xu_open_dataset_mock.assert_called_once_with("path/to/mesh.nc")
    assert mesh is ugrid_mock


def test_parse_region_mesh_dataset(mocker: MockerFixture):
    ugrid_mock = mocker.Mock(spec_set=xu.UgridDataset)
    mesh = parse_region_mesh({"mesh": ugrid_mock})
    assert mesh is ugrid_mock

    ugrid_mock = mocker.Mock(spec_set=xu.UgridDataArray)
    mesh = parse_region_mesh({"mesh": ugrid_mock})
    assert mesh is ugrid_mock


def test_parse_region_mesh_wrong_type():
    with pytest.raises(ValueError, match="Unrecognized type"):
        parse_region_mesh({"mesh": 123})
