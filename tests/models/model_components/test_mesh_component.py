import logging
from unittest.mock import create_autospec, patch

import pytest
import xarray as xr
import xugrid as xu

from hydromt.models.components.mesh import MeshComponent, _check_UGrid


def test_check_UGrid():
    data = xr.DataArray()
    with pytest.raises(
        ValueError,
        match="New mesh data in set_mesh should be of type xu.UgridDataArray"
        " or xu.UgridDataset",
    ):
        _check_UGrid(data=data, name=None)
    data = create_autospec(xu.UgridDataArray)
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


@patch.object(MeshComponent, "_grid_is_equal")
def test_add_mesh(mock_grid_is_equal, mock_model, caplog):
    mesh_component = MeshComponent(mock_model)
    data = xu.data.elevation_nl().to_dataset()
    with pytest.raises(ValueError, match="Data should have CRS."):
        mesh_component._add_mesh(data=data, grid_name="", overwrite_grid=False)
    data.grid.crs = 28992
    mesh_component._data = data
    data4326 = xu.data.elevation_nl().to_dataset()
    data4326.grid.crs = 4326
    with pytest.raises(
        ValueError, match="Data and self.data should have the same CRS."
    ):
        mesh_component._add_mesh(data=data4326, grid_name="", overwrite_grid=False)
    grid_name = "mesh2d"
    mock_grid_is_equal.return_value = False
    with pytest.raises(
        ValueError,
        match=f"Grid {grid_name} already exists in mesh"
        " and has a different topology. "
        "Use overwrite_grid=True to overwrite the grid"
        " topology and related data.",
    ):
        mesh_component._add_mesh(data=data, grid_name=grid_name, overwrite_grid=False)
    caplog.set_level(logging.WARNING)
    mesh_component._data = data
    mesh_component._add_mesh(data=data, grid_name="mesh2d", overwrite_grid=True)
    assert (
        f"Overwriting grid {grid_name} and the corresponding data variables in mesh."
        in caplog.text
    )


def test_set_raises_errors(mock_model):
    mesh_component = MeshComponent(mock_model)
    with pytest.raises(
        ValueError,
        match=(
            "New mesh data in set_mesh should be of"
            " type xu.UgridDataArray or xu.UgridDataset"
        ),
    ):
        mesh_component.set(xr.Dataset())

    data_no_name = create_autospec(xu.UgridDataArray)
    data_no_name.name = None
    with pytest.raises(
        ValueError,
        match=f"Cannot set mesh from {str(type(data_no_name).__name__)} without a name.",
    ):
        mesh_component.set(data_no_name)
    data = create_autospec(xu.UgridDataset)
    data.name = "fakedata"
    data.ugrid.grids = [1, 2]
    with pytest.raises(
        ValueError,
        match="set_mesh methods only supports adding data to one grid at a time.",
    ):
        mesh_component.set(data)

    data = create_autospec(xu.UgridDataset)
    data.ugrid.grid.crs = None
    with patch("hydromt.models.components.mesh.MeshComponent.data", None):
        with pytest.raises(ValueError, match="Data should have CRS."):
            mesh_component.set(data)

    with patch("hydromt.models.components.mesh.MeshComponent.data", data):
        with patch("hydromt.models.components.mesh.MeshComponent.crs", 4326):
            with pytest.raises(
                ValueError, match="Data and self.data should have the same CRS."
            ):
                mesh_component.set(data)
            data.ugrid.grid.crs = 4326
            data.ugrid.grid.name = "fake_grid"
            with patch(
                "hydromt.models.components.mesh.MeshComponent.mesh_names",
                data.ugrid.grid.name,
            ):
                with pytest.raises(
                    ValueError,
                    match=f"Grid {data.ugrid.grid.name} already exists in mesh"
                    " and has a different topology. "
                    "Use overwrite_grid=True to overwrite the grid"
                    " topology and related data.",
                ):
                    mesh_component.set(data)


def test_create():
    pass


def test_write():
    pass


def test_read():
    pass


def test_properties():
    pass


def test_get_mesh():
    pass


def test_add_data_from_rasterdataset():
    pass


def test_add_data_from_raster_reclass():
    pass
