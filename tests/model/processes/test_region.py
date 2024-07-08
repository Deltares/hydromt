import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import xugrid as xu
from pytest_mock import MockerFixture
from shapely import box

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


def test_region_from_file(tmp_path: Path, world: gpd.GeoDataFrame):
    uri_gdf = tmp_path / "world.gpkg"
    world.to_file(uri_gdf, driver="GPKG")
    gpd.testing.assert_geodataframe_equal(gpd.read_file(uri_gdf), world)
    region = parse_region_geom({"geom": uri_gdf}, crs=None, data_catalog=DataCatalog())
    gpd.testing.assert_geodataframe_equal(world, region)


def test_geom_from_cat(tmpdir, world):
    uri_gdf = str(tmpdir.join("world.geojson"))
    world.to_file(uri_gdf, driver="GeoJSON")
    cat = DataCatalog()
    cat.from_dict(
        {
            "world": {
                "uri": uri_gdf,
                "data_type": "GeoDataFrame",
                "driver": "pyogrio",
            },
        }
    )
    region = parse_region_geom({"geom": "world"}, crs=None, data_catalog=cat)
    assert isinstance(region, gpd.GeoDataFrame)


def test_geom_from_points_fails(geodf):
    region = {"geom": geodf}
    with pytest.raises(ValueError, match=r"Region value.*"):
        region = parse_region_geom(region=region, crs=None)


def test_region_from_grid(rioda):
    region = parse_region_grid({"grid": rioda}, data_catalog=None)
    xr.testing.assert_equal(region, rioda)


def test_region_from_grid_file(tmp_path: Path, rioda: xr.Dataset):
    uri_grid = str(tmp_path / "grid.tif")
    rioda.raster.to_raster(uri_grid)
    region = parse_region_grid({"grid": uri_grid}, data_catalog=DataCatalog())
    xr.testing.assert_equal(region, rioda)


def test_region_from_grid_cat(tmp_path: Path, rioda: xr.Dataset):
    uri_grid = str(tmp_path / "grid.tif")
    rioda.raster.to_raster(uri_grid)
    cat = DataCatalog()
    cat.from_dict(
        {
            "grid": {
                "uri": uri_grid,
                "data_type": "RasterDataset",
                "driver": "rasterio",
            },
        }
    )
    region = parse_region_grid({"grid": "grid"}, data_catalog=cat)
    xr.testing.assert_equal(region, rioda)


def test_region_from_basin_ids():
    basin_ids = [210000018, 210000030, 210000039]
    region = {"basin": basin_ids}
    region = parse_region_basin(
        region,
        data_catalog=DataCatalog(),
        hydrography_path="merit_hydro",
        basin_index_path="merit_hydro_index",
    )
    assert set(map(lambda x: int(x), region["value"])) == set(basin_ids)


def test_region_from_basin_id():
    basin_id = 210000018
    region = {"basin": basin_id}
    region = parse_region_basin(
        region,
        data_catalog=DataCatalog(),
        hydrography_path="merit_hydro",
        basin_index_path="merit_hydro_index",
    )
    assert set(region["value"]) == {basin_id}


def test_region_from_subbasin():
    region_dict = {
        "subbasin": [12.3, 46.2],
        "uparea": 5.0,
        "bounds": [12.0, 46.0, 12.5, 46.5],
    }
    region: gpd.GeoDataFrame = parse_region_basin(
        region_dict,
        data_catalog=DataCatalog(),
        hydrography_path="merit_hydro",
        basin_index_path="merit_hydro_index",
    )
    # area should have something to do with our region
    assert box(*region_dict["bounds"]).intersects(region.loc[0].geometry)
    assert region.shape == (1, 2)


def test_region_from_basin_xy():
    region = {"basin": [[12.0, 46.0], [12.3, 46.5]]}
    region = parse_region_basin(
        region,
        data_catalog=DataCatalog(),
        hydrography_path="merit_hydro",
        basin_index_path="merit_hydro_index",
    )
    assert region.shape == (4, 2)


def test_region_from_inter_basin():
    region = {
        "interbasin": [12.0, 46.0, 12.5, 46.5],
        "xy": [12.0, 46.0],
    }
    region = parse_region_basin(
        region,
        data_catalog=DataCatalog(),
        hydrography_path="merit_hydro",
        basin_index_path="merit_hydro_index",
    )
    assert region.shape == (1, 2)


def test_raise_wrong_region_value_for_interbasin():
    region = {"interbasin": [12.0, 46.0]}
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"Region value '[12.0, 46.0]' for kind=interbasin not understood, provide one of geom,bbox"
        ),
    ):
        region = parse_region_basin(
            region,
            data_catalog=DataCatalog(),
            hydrography_path="merit_hydro",
            basin_index_path="merit_hydro_index",
        )


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


def test_region_mesh(griduda):
    mesh = parse_region_mesh({"mesh": griduda})
    assert np.all(mesh == griduda)


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
