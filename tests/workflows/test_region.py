import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from pytest_mock import MockerFixture

from hydromt import DataCatalog, raster
from hydromt.models.model import Model
from hydromt.workflows.basin_mask import _check_size
from hydromt.workflows.region import _parse_region_value, parse_region


def test_region_from_geom(world):
    region = parse_region({"geom": world})
    assert world is region


@pytest.mark.skip("Needs new driver implementation")
def test_region_from_file(tmpdir, world):
    fn_gdf = str(tmpdir.join("world.geojson"))
    world.to_file(fn_gdf, driver="GeoJSON")
    region = parse_region({"geom": fn_gdf}, data_catalog=DataCatalog())
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
    region = parse_region({"geom": "world"}, data_catalog=cat)
    assert isinstance(region["geom"], gpd.GeoDataFrame)


def test_geom_from_points_fails(geodf):
    region = {"geom": geodf}
    with pytest.raises(ValueError, match=r"Region value.*"):
        region = parse_region(region)


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_grid(rioda):
    region = parse_region({"grid": rioda})
    assert isinstance(region, xr.DataArray)


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_grid_file(tmpdir, rioda):
    fn_grid = str(tmpdir.join("grid.tif"))
    rioda.raster.to_raster(fn_grid)
    region = parse_region({"grid": fn_grid})
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
    region = parse_region({"grid": "grid"}, data_catalog=cat)
    assert isinstance(region["grid"], xr.DataArray)


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_basin_ids():
    region = {"basin": [1001, 1002, 1003, 1004, 1005]}
    region = parse_region(
        region,
        data_catalog=DataCatalog(),
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
    )
    assert region.get("basid") == [1001, 1002, 1003, 1004, 1005]


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_basin_id():
    region = {"basin": 101}
    region = parse_region(
        region,
        data_catalog=DataCatalog(),
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
    )
    assert region.get("basid") == 101


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_subbasin():
    region = {"subbasin": [1.0, -1.0], "uparea": 5.0, "bounds": [0.0, -5.0, 3.0, 0.0]}
    region = parse_region(
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
    region = parse_region(
        region,
        data_catalog=DataCatalog(),
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
    )
    assert "xy" in region


@pytest.mark.skip(reason="Needs Rasterdataset impl")
def test_region_from_inter_basin(geodf):
    region = {"interbasin": geodf}
    region = parse_region(
        region,
        data_catalog=DataCatalog(),
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
    )
    assert "xy" in region


def test_region_from_model(tmpdir, world, mocker: MockerFixture):
    model = mocker.Mock(spec=Model, region=world)
    plugins = mocker.patch("hydromt.workflows.region.PLUGINS")
    plugins.model_plugins = {"Model": mocker.Mock(return_value=model)}
    region = {Model.__name__: tmpdir}
    region = parse_region(region)
    assert region is world


@pytest.mark.skip(reason="MODEL import does not work yet")
def test_region_from_unknown_errors():
    # model
    region = {"region": [0.0, -1.0]}
    with pytest.raises(ValueError, match=r"Region key .* not understood.*"):
        parse_region(region)


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
