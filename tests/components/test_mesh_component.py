import logging
import os
from os.path import dirname, join
from pathlib import Path
from typing import cast

import geopandas as gpd
import pytest
import xarray as xr
import xugrid as xu
from pytest_mock import MockerFixture

from hydromt.model import Model
from hydromt.model.components.mesh import (
    MeshComponent,
)
from hydromt.model.processes.mesh import create_mesh2d_from_region
from hydromt.model.root import ModelRoot


def test_check_ugrid(mocker: MockerFixture):
    data = xr.DataArray()
    with pytest.raises(
        ValueError,
        match="New mesh data in set_mesh should be of type xu.UgridDataArray"
        " or xu.UgridDataset",
    ):
        MeshComponent._check_ugrid(data=data, name=None)
    data = mocker.create_autospec(xu.UgridDataArray)
    data.name = None
    with pytest.raises(
        ValueError,
        match=f"Cannot set mesh from {str(type(data).__name__)} without a name.",
    ):
        MeshComponent._check_ugrid(data=data, name=None)

    data = xu.data.elevation_nl()
    dataset = MeshComponent._check_ugrid(data=data, name="new_dataset")
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


def test_add_mesh_logging(
    mocker: MockerFixture, mock_model, caplog: pytest.LogCaptureFixture
):
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
    mock_check_ugrid = mocker.patch(
        "hydromt.model.components.mesh.MeshComponent._check_ugrid"
    )
    mock_check_ugrid.return_value = data
    with pytest.raises(
        ValueError,
        match="set_mesh methods only supports adding data to one grid at a time.",
    ):
        mesh_component.set(data=data)


def test_model_mesh_sets_correctly(tmp_path: Path):
    m = Model(root=tmp_path, mode="r+")
    m.add_component("mesh", MeshComponent(m))
    component = cast(MeshComponent, m.mesh)
    uds = xu.data.elevation_nl().to_dataset()
    uds.grid.crs = 28992
    component.set(data=uds)
    assert component.data == uds


def test_write(mock_model, caplog: pytest.LogCaptureFixture, tmp_path: Path):
    mesh_component = MeshComponent(mock_model)
    mesh_component.root.is_reading_mode.return_value = False
    mock_model.components["mesh"] = mesh_component
    mock_model.name = "foo"
    caplog.set_level(logging.INFO)
    mesh_component.write()
    assert "foo.mesh: No mesh data found, skip writing." in caplog.text
    mock_model.root = ModelRoot(path=tmp_path, mode="r")
    mesh_component._data = xu.data.elevation_nl().to_dataset()
    with pytest.raises(IOError, match="Model opened in read-only mode"):
        mesh_component.write()
    mock_model.root = ModelRoot(path=tmp_path, mode="w")
    write_path = "mesh/fake_mesh.nc"
    mesh_component._data.grid.crs = 28992
    mesh_component.write(filename=write_path)
    file_path = tmp_path / write_path
    assert file_path.parent.is_dir()
    assert f"foo.mesh: Writing mesh to {file_path}" in caplog.text
    assert file_path.is_file()
    ds = xr.open_dataset(file_path)
    assert "elevation" in ds.data_vars
    assert "28992" in ds.spatial_ref.crs_wkt


def test_model_mesh_read_plus(tmp_path: Path):
    m = Model(root=tmp_path, mode="w")
    m.add_component("mesh", MeshComponent(m))
    data = xu.data.elevation_nl().to_dataset()
    data.grid.crs = 28992
    cast(MeshComponent, m.mesh).set(data=data, grid_name="elevation_mesh")
    m.write()
    m2 = Model(root=tmp_path, mode="r+")
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


@pytest.mark.integration
def test_read(mock_model, caplog: pytest.LogCaptureFixture, tmp_path: Path, griduda):
    mesh_component = MeshComponent(mock_model)
    mesh_component.model.root = ModelRoot(tmp_path, mode="w")
    with pytest.raises(IOError, match="Model opened in write-only mode"):
        mesh_component.read()
    mesh_path = "test/test_mesh.nc"
    file_dir = join(mesh_component.model.root.path, dirname(mesh_path))
    os.makedirs(file_dir)
    data = griduda.ugrid.to_dataset()
    data.to_netcdf(join(mesh_component.model.root.path, mesh_path))
    mock_model.root = ModelRoot(tmp_path, mode="r+")
    with pytest.raises(
        ValueError, match="no crs is found in the file nor passed to the reader."
    ):
        mesh_component.read(filename=mesh_path)
    caplog.set_level(level=logging.INFO)
    mesh_component.read(filename=mesh_path, crs=4326)
    assert "no crs is found in the file, assigning from user input." in caplog.text
    assert mesh_component._data.ugrid.crs["mesh2d"].to_epsg() == 4326


def test_model_mesh_workflow(tmp_path: Path):
    m = Model(root=tmp_path, mode="r+")
    m.add_component("mesh", MeshComponent(m))
    component = cast(MeshComponent, m.mesh)
    region = {
        "bbox": [11.949099, 45.9722, 12.004855, 45.998441]
    }  # small area in Piave basin
    crs = 4326
    res = 0.001
    mesh = create_mesh2d_from_region(region=region, res=res, crs=crs)
    component.set(data=mesh)

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
