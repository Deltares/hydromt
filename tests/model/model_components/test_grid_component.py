import numpy as np
import pytest
import xarray as xr

from hydromt.models.v1.grid import GridComponent


def test_set_GridComponent(hydds, tmp_dir, rioda):
    gridmodel_component = GridComponent(root=tmp_dir)
    # Test setting xr.Dataset
    gridmodel_component.set(data=hydds)
    assert len(gridmodel_component._data) > 0
    assert isinstance(gridmodel_component._data, xr.Dataset)
    # Test setting xr.DataArray
    data_array = hydds.to_array()
    gridmodel_component.set(data=data_array, name="data_array")
    assert "data_array" in gridmodel_component._data.data_vars.keys()
    assert len(gridmodel_component._data.data_vars) == 3
    # Test setting nameless data array
    data_array.name = None
    with pytest.raises(
        ValueError,
        match=f"Unable to set {type(data_array).__name__} data without a name",
    ):
        gridmodel_component.set(data=data_array)
    # Test setting np.ndarray of different shape
    ndarray = np.random.rand(4, 5)
    with pytest.raises(ValueError, match="Shape of data and grid maps do not match"):
        gridmodel_component.set(ndarray, name="ndarray")
