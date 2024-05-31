import logging
import os
import re
from os.path import abspath, dirname, isdir, isfile, join
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xugrid as xu
from pytest_mock import MockerFixture

from hydromt.model import Model
from hydromt.model.components.mesh import MeshComponent, _check_UGrid
from hydromt.model.root import ModelRoot

DATADIR = join(dirname(dirname(abspath(__file__))), "data")


def test_check_UGrid(mocker: MockerFixture):
    data = xr.DataArray()
    with pytest.raises(
        ValueError,
        match="New mesh data in set_mesh should be of type xu.UgridDataArray"
        " or xu.UgridDataset",
    ):
        _check_UGrid(data=data, name=None)
    data = mocker.create_autospec(xu.UgridDataArray)
    data.name = None
    with pytest.raises(
        ValueError,
        match=f"Cannot set mesh from {str(type(data).__name__)} without a name.",
    ):
        _check_UGrid(data=data, name=None)

    data = xu.data.elevation_nl()  # TODO: maybe generate data instead
    dataset = _check_UGrid(data=data, name="new_dataset")
    assert isinstance(dataset, xu.UgridDataset)
    assert "new_dataset" in dataset.data_vars.keys()


def test_add_mesh_errors(mock_model, mocker: MockerFixture):
    mesh_component = MeshComponent(mock_model)
    mesh_component.root.is_reading_mode.return_value = False
    data = xu.data.elevation_nl().to_dataset()
    with pytest.raises(ValueError, match="Data should have CRS."):
        mesh_component._add_mesh(data=data, grid_name="", overwrite_grid=False)
    data.grid.crs = 28992
    mesh_component._data = data
    data4326 = xu.data.elevation_nl().to_dataset()
    data4326.grid.crs = 4326
    with pytest.raises(ValueError, match="Data and Mesh should have the same CRS."):
        mesh_component._add_mesh(data=data4326, grid_name="", overwrite_grid=False)
    grid_name = "mesh2d"
    mock_grid_is_equal = mocker.patch.object(MeshComponent, "_grid_is_equal")
    mock_grid_is_equal.return_value = False
    with pytest.raises(
        ValueError,
        match=f"Grid {grid_name} already exists in mesh"
        " and has a different topology. "
        "Use overwrite_grid=True to overwrite the grid"
        " topology and related data.",
    ):
        mesh_component._add_mesh(data=data, grid_name=grid_name, overwrite_grid=False)


def test_add_mesh_logging(mocker: MockerFixture, mock_model, caplog):
    mesh_component = MeshComponent(mock_model)
    data = xu.data.elevation_nl().to_dataset()
    mock_grid_is_equal = mocker.patch.object(MeshComponent, "_grid_is_equal")
    mock_grid_is_equal.return_value = False
    caplog.set_level(logging.WARNING)
    mesh_component._data = data
    grid_name = "mesh2d"
    mesh_component._add_mesh(data=data, grid_name=grid_name, overwrite_grid=True)
    assert (
        f"Overwriting grid {grid_name} and the corresponding data variables in mesh."
        in caplog.text
    )
    mock_grid_is_equal.return_value = True
    mesh_component._data = data
    mesh_component._add_mesh(data=data, grid_name="mesh2d", overwrite_grid=False)
    assert "Replacing mesh parameter: elevation" in caplog.text


def test_add_mesh(mock_model):
    mesh_component = MeshComponent(mock_model)
    data = xu.data.elevation_nl().to_dataset()
    data.grid.crs = 28992
    mesh_component._data = data
    mesh_component._add_mesh(data=data, grid_name="", overwrite_grid=False)
    assert mesh_component.crs == 28992
    assert data.grid.name in mesh_component.mesh_names


def test_set_raises_errors(mocker: MockerFixture, mock_model):
    mesh_component = MeshComponent(mock_model)
    data = mocker.create_autospec(xu.UgridDataset)
    data.name = "fakedata"
    data.ugrid.grids = [1, 2]
    mock_check_Ugrid = mocker.patch("hydromt.model.components.mesh._check_UGrid")
    mock_check_Ugrid.return_value = data
    with pytest.raises(
        ValueError,
        match="set_mesh methods only supports adding data to one grid at a time.",
    ):
        mesh_component.set(data=data)


def test_model_mesh_sets_correctly(tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("mesh", MeshComponent(m))
    component = cast(MeshComponent, m.mesh)
    uds = xu.data.elevation_nl().to_dataset()
    uds.grid.crs = 28992
    component.set(data=uds)
    assert component.data == uds


def test_create(mock_model, mocker: MockerFixture):
    mesh_component = MeshComponent(mock_model)
    mesh_component.root.is_reading_mode.return_value = False
    region = {"bbox": [-1, -1, 1, 1]}
    res = 20
    crs = 28992
    test_data = xu.data.elevation_nl().to_dataset()
    test_data.grid.crs = crs
    mock_create_mesh2d = mocker.patch(
        "hydromt.model.components.mesh.create_mesh2d_from_region"
    )
    mock_create_mesh2d.return_value = test_data
    mesh_component.create_2d_from_region(region=region, res=res, crs=crs)
    mock_create_mesh2d.assert_called_once()
    assert mesh_component.data == test_data


def test_write(mock_model, caplog, tmpdir):
    mesh_component = MeshComponent(mock_model)
    caplog.set_level(logging.DEBUG)
    mesh_component.root.is_reading_mode.return_value = False
    mesh_component.write()
    assert "No mesh data found, skip writing." in caplog.text
    mock_model.root = ModelRoot(path=tmpdir, mode="r")
    mesh_component._data = xu.data.elevation_nl().to_dataset()
    with pytest.raises(IOError, match="Model opened in read-only mode"):
        mesh_component.write()
    mock_model.root = ModelRoot(path=tmpdir, mode="w")
    fn = "mesh/fake_mesh.nc"
    mesh_component._data.grid.crs = 28992
    mesh_component.write(filename=fn)
    file_dir = join(mesh_component.root.path, dirname(fn))
    file_path = join(tmpdir, fn)
    assert isdir(file_dir)
    assert f"Writing file {fn}" in caplog.text
    assert isfile(file_path)
    ds = xr.open_dataset(file_path)
    assert "elevation" in ds.data_vars
    assert "28992" in ds.spatial_ref.crs_wkt


def test_model_mesh_read_plus(tmpdir: Path):
    m = Model(root=str(tmpdir), mode="w")
    m.add_component("mesh", MeshComponent(m))
    data = xu.data.elevation_nl().to_dataset()
    data.grid.crs = 28992
    cast(MeshComponent, m.mesh).set(data=data, grid_name="elevation_mesh")
    m.write()
    m2 = Model(root=str(tmpdir), mode="r+")
    m2.add_component("mesh", MeshComponent(m2))
    data = xu.data.elevation_nl().to_dataset()
    data.grid.crs = 28992
    data = data.rename({"elevation": "elevation_v2"})
    mesh2_component = cast(MeshComponent, m2.mesh)
    mesh2_component.set(data=data, grid_name="elevation_mesh")
    assert "elevation_v2" in mesh2_component.data.data_vars
    assert "elevation" in mesh2_component.data.data_vars
    assert mesh2_component.crs.to_epsg() == 28992


def test_properties(mock_model):
    mesh_component = MeshComponent(mock_model)
    data = xu.data.adh_san_diego()
    # Test crs
    data.ugrid.set_crs(4326)
    mesh_component._data = data
    assert mesh_component.crs == 4326
    # Test bounds
    assert mesh_component.bounds == data.ugrid.bounds
    # Test mesh_names
    mesh_names = mesh_component.mesh_names
    assert len(mesh_names) == 1
    assert "mesh2d" in mesh_names
    # Test mesh_grids
    mesh_grids = mesh_component.mesh_grids
    assert len(mesh_grids) == 1
    assert isinstance(mesh_grids["mesh2d"], xu.Ugrid2d)
    # Test mesh_datasets
    mesh_datasets = mesh_component.mesh_datasets
    assert len(mesh_datasets) == 1
    assert isinstance(mesh_datasets["mesh2d"], xu.UgridDataset)
    # Test mesh_gdf
    mesh_gdf = mesh_component.mesh_gdf
    assert len(mesh_gdf) == 1
    assert isinstance(mesh_gdf["mesh2d"], gpd.GeoDataFrame)


def test_get_mesh(mock_model):
    mesh_component = MeshComponent(mock_model)
    mesh_component.root.is_reading_mode.return_value = False
    with pytest.raises(ValueError, match="Mesh is not set, please use set_mesh first."):
        mesh_component.get_mesh(grid_name="")
    mesh_component._data = xu.data.elevation_nl().to_dataset()
    grid_name = "test_grid"
    with pytest.raises(ValueError, match=f"Grid {grid_name} not found in mesh."):
        mesh_component.get_mesh(grid_name=grid_name)

    mesh = mesh_component.get_mesh(grid_name="mesh2d")
    assert isinstance(mesh, xu.Ugrid2d)
    mesh_component._data.grid.crs = 4326
    mesh = mesh_component.get_mesh(grid_name="mesh2d", include_data=True)
    assert isinstance(mesh, xu.UgridDataset)


def test_add_2d_data_from_rasterdataset(mock_model, caplog, mocker: MockerFixture):
    mesh_component = MeshComponent(mock_model)
    mesh_component.data_catalog.get_rasterdataset.return_value = xr.Dataset()
    mock_data = xu.data.elevation_nl().to_dataset()
    mock_data.grid.set_crs(28992)
    mesh_component.set(mock_data)
    grid_name = "test_grid"
    caplog.set_level(level=logging.INFO)
    with pytest.raises(
        ValueError,
        match=re.escape(f"Grid {grid_name} not found in mesh."),
    ):
        mesh_component.add_2d_data_from_rasterdataset(
            raster_filename="mock_raster", grid_name=grid_name
        )

    mock_mesh2d_from_rasterdataset = mocker.patch(
        "hydromt.model.components.mesh.mesh2d_from_rasterdataset"
    )
    mock_mesh2d_from_rasterdataset.return_value = mock_data

    data_vars = mesh_component.add_2d_data_from_rasterdataset(
        raster_filename="vito", grid_name="mesh2d", resampling_method="mode"
    )
    assert "Preparing mesh data from raster source vito" in caplog.text
    assert all([var in mock_data.data_vars.keys() for var in data_vars])
    assert mesh_component.data == mock_data
    assert "mesh2d" in mesh_component.mesh_names


def test_add_2d_data_from_raster_reclass(mock_model, caplog, mocker: MockerFixture):
    mesh_component = MeshComponent(mock_model)
    mesh_component.data_catalog.get_rasterdataset.return_value = xr.Dataset()
    mock_data = xu.data.elevation_nl().to_dataset()
    mock_data.grid.set_crs(28992)
    mesh_component.set(mock_data)
    grid_name = "test_grid"
    caplog.set_level(level=logging.INFO)
    with pytest.raises(
        ValueError,
        match=re.escape(f"Grid {grid_name} not found in mesh."),
    ):
        mesh_component.add_2d_data_from_raster_reclass(
            raster_filename="mock_raster",
            grid_name=grid_name,
            reclass_table_filename="mock_reclass_table",
            reclass_variables=["landuse", "roughness_manning"],
        )
    raster_fn = "mock_raster"
    with pytest.raises(
        ValueError,
        match=f"raster_filename {raster_fn} should be a single variable raster. "
        "Please select one using the 'variable' argument",
    ):
        mesh_component.add_2d_data_from_raster_reclass(
            raster_filename=raster_fn,
            reclass_table_filename="reclass_table",
            grid_name="mesh2d",
            reclass_variables=["landuse", "roughness_manning"],
        )

    mesh_component.data_catalog.get_rasterdataset.return_value = xr.DataArray()
    mesh_component.data_catalog.get_dataframe.return_value = pd.DataFrame()
    mock_mesh2d_from_rasterdataset = mocker.patch(
        "hydromt.model.components.mesh.mesh2d_from_raster_reclass"
    )

    mock_mesh2d_from_rasterdataset.return_value = mock_data
    data_vars = mesh_component.add_2d_data_from_raster_reclass(
        raster_filename="vito",
        grid_name="mesh2d",
        resampling_method="mode",
        reclass_table_filename="vito_mapping",
        reclass_variables=["landuse", "roughness_manning"],
    )
    assert (
        "Preparing mesh data by reclassifying the data in vito based on vito_mapping"
        in caplog.text
    )
    assert all([var in mock_data.data_vars.keys() for var in data_vars])
    assert mesh_component.data == mock_data
    assert "mesh2d" in mesh_component.mesh_names


@pytest.mark.integration()
def test_read(mock_model, caplog, tmpdir, griduda):
    mesh_component = MeshComponent(mock_model)
    mesh_component.model.root = ModelRoot(tmpdir, mode="w")
    with pytest.raises(IOError, match="Model opened in write-only mode"):
        mesh_component.read()
    fn = "test/test_mesh.nc"
    file_dir = join(mesh_component.model.root.path, dirname(fn))
    os.makedirs(file_dir)
    data = griduda.ugrid.to_dataset()
    data.to_netcdf(join(mesh_component.model.root.path, fn))
    mock_model.root = ModelRoot(tmpdir, mode="r+")
    with pytest.raises(
        ValueError, match="no crs is found in the file nor passed to the reader."
    ):
        mesh_component.read(filename=fn)
    caplog.set_level(level=logging.INFO)
    mesh_component.read(filename=fn, crs=4326)
    assert "no crs is found in the file, assigning from user input." in caplog.text
    assert mesh_component._data.ugrid.crs["mesh2d"].to_epsg() == 4326


def test_model_mesh_workflow(tmpdir: Path):
    m = Model(root=str(tmpdir), mode="r+")
    m.add_component("mesh", MeshComponent(m))
    component = cast(MeshComponent, m.mesh)
    region = {
        "bbox": [11.949099, 45.9722, 12.004855, 45.998441]
    }  # small area in Piave basin
    crs = 4326
    res = 0.001
    mesh = component.create_2d_from_region(region=region, res=res, crs=crs)
    assert component.data.grid.crs == crs
    # clear empty mesh dataset
    mesh._data = None
    # Test with sample data
    data = xu.data.elevation_nl().to_dataset()
    data.grid.crs = 28992
    component.set(data=data, grid_name="elevation_mesh")
    assert "elevation_mesh" in component.mesh_names
    assert component.data == data
    component.write()
    component._data = None
    component.read()
    assert component.data == data


@pytest.mark.integration()
def test_mesh_with_model(griduda, world, tmpdir):
    dc_param_fn = join(DATADIR, "parameters_data.yml")
    root = join(tmpdir, "mesh_component1")
    model = Model(root=root, data_libs=["artifact_data", dc_param_fn])
    mesh_component = MeshComponent(model=model)
    model.add_component(name="mesh", component=mesh_component)
    region = {"geom": world[world.name == "Italy"]}
    model.mesh.create_2d_from_region(region, res=10000, crs=3857, grid_name="mesh2d")
    assert model.mesh.region.crs.to_epsg() == 3857

    region = {"mesh": griduda}
    model1 = Model(root=root, data_libs=["artifact_data", dc_param_fn])
    mesh_component = MeshComponent(model=model1)
    model1.add_component(name="mesh", component=mesh_component)
    model1.mesh.create_2d_from_region(region, grid_name="mesh2d")
    model1.mesh.add_2d_data_from_rasterdataset(
        "vito", grid_name="mesh2d", resampling_method="mode"
    )
    assert "vito" in model1.mesh.data.data_vars

    model1.mesh.add_2d_data_from_raster_reclass(
        raster_filename="vito",
        reclass_table_filename="vito_mapping",
        reclass_variables=["landuse", "roughness_manning"],
        resampling_method=["mode", "centroid"],
        grid_name="mesh2d",
    )
    ds_mesh2d = model1.mesh.get_mesh("mesh2d", include_data=True)

    assert "vito" in ds_mesh2d
    assert "roughness_manning" in model1.mesh.data.data_vars
    assert np.all(model1.mesh.data["landuse"].values == model1.mesh.data["vito"].values)

    # write model
    model1.write()

    # Read model
    written_model = Model(root=root, mode="r")
    mesh_component = MeshComponent(model=written_model)
    written_model.add_component(name="mesh", component=mesh_component)
    written_model.read()
    variables = ["roughness_manning", "vito", "landuse"]
    assert all(
        [data_var in variables for data_var in written_model.mesh.data.data_vars]
    )
