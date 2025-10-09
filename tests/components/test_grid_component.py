import logging
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from pytest_mock import MockerFixture

from hydromt.model.components.grid import (
    GridComponent,
)


def test_set_dataset(mock_model, hydds):
    grid_component = GridComponent(model=mock_model)
    grid_component.set(data=hydds)
    assert len(grid_component.data) > 0
    assert isinstance(grid_component.data, xr.Dataset)


def test_set_dataarray(mock_model, hydds):
    grid_component = GridComponent(model=mock_model)
    data_array = hydds.to_array()
    grid_component.set(data=data_array, name="data_array")
    assert "data_array" in grid_component.data.data_vars.keys()
    assert len(grid_component.data.data_vars) == 1


def test_set_raise_errors(mock_model, hydds):
    grid_component = GridComponent(model=mock_model)
    # Test setting nameless data array
    data_array = hydds.to_array()
    with pytest.raises(
        ValueError,
        match=f"Unable to set {type(data_array).__name__} data without a name",
    ):
        grid_component.set(data=data_array)
    # Test setting np.ndarray of different shape
    grid_component.set(data=data_array, name="data_array")
    ndarray = np.random.rand(4, 5)
    with pytest.raises(ValueError, match="Shape of data and grid maps do not match"):
        grid_component.set(ndarray, name="ndarray")


def test_set_empty_ndarray_raises(mock_model):
    grid = GridComponent(model=mock_model)
    with pytest.raises(ValueError, match="empty array"):
        grid.set(np.array([]), name="empty_layer")


def test_write(
    mock_model, tmp_path: Path, caplog: pytest.LogCaptureFixture, mocker: MockerFixture
):
    grid_component = GridComponent(model=mock_model)
    mock_model.components["grid"] = grid_component
    mock_model.name = "foo"
    # Test skipping writing when no grid data has been set
    with caplog.at_level(logging.INFO):
        grid_component.write()
    assert "foo.grid: No grid data found, skip writing" in caplog.text
    # Test raise IOerror when model is in read only mode
    mock_model.root.set(path=tmp_path, mode="r")
    grid_component = GridComponent(model=mock_model)
    mocker.patch.object(GridComponent, "data", ["test"])
    with pytest.raises(IOError, match="Model opened in read-only mode"):
        grid_component.write()


def test_read(tmp_path: Path, mock_model, hydds, mocker: MockerFixture):
    # Test for raising IOError when model is in writing mode
    grid_component = GridComponent(model=mock_model)
    with pytest.raises(IOError, match="Model opened in write-only mode"):
        grid_component.read()
    mock_model.root.set(path=tmp_path, mode="r+")
    grid_component = GridComponent(model=mock_model)
    mocker.patch("hydromt.model.components.grid.open_ncs", return_value={"grid": hydds})
    grid_component.read()
    assert grid_component.data == hydds


def test_properties(caplog: pytest.LogCaptureFixture, demda, mock_model):
    grid_component = GridComponent(mock_model)
    # Test properties on empty grid
    caplog.set_level(logging.WARNING)
    res = grid_component.res
    assert "No grid data found for deriving resolution" in caplog.text
    transform = grid_component.transform
    assert "No grid data found for deriving transform" in caplog.text
    crs = grid_component.crs
    assert "No grid data found for deriving resolution" in caplog.text
    bounds = grid_component.bounds
    assert "No grid data found for deriving bounds" in caplog.text
    region = grid_component.region
    assert "No grid data found for deriving region" in caplog.text
    assert all(
        props is None for props in [res, transform, crs, bounds, region]
    )  # Ruff complains if prop vars are unused

    grid_component._data = demda
    assert grid_component.res == demda.raster.res
    assert grid_component.transform == demda.raster.transform
    assert grid_component.crs == demda.raster.crs
    assert grid_component.bounds == demda.raster.bounds

    region = grid_component.region
    assert isinstance(region, gpd.GeoDataFrame)
    assert all(region.bounds == demda.raster.bounds)


def test_initialize_grid(mock_model, tmp_path: Path):
    mock_model.root.set(path=tmp_path, mode="r")
    grid_component = GridComponent(mock_model)
    grid_component.read = MagicMock()
    grid_component._initialize_grid()
    assert isinstance(grid_component._data, xr.Dataset)
    assert grid_component.read.called


@pytest.fixture
def base_grid():
    return xr.Dataset(
        data_vars={"base": (["y", "x"], np.ones((5, 5)))},
        coords={"x": np.arange(5), "y": np.arange(5)},
    )


@pytest.fixture
def grid_component(mock_model, base_grid):
    """Fixture to create a GridComponent with a base grid."""
    comp = GridComponent(model=mock_model)
    comp._data = base_grid.copy(deep=True)
    return comp


def test_set_no_mask(grid_component: GridComponent):
    grid_size = 5
    target = 7
    new_data = xr.DataArray(
        np.full((grid_size, grid_size), target), dims=("y", "x"), name="layer1"
    )
    grid_component.set(new_data)
    assert "layer1" in grid_component._data
    assert (grid_component._data["layer1"] == target).all()


def test_set_mask_array(grid_component: GridComponent):
    grid_size = 5
    nodata = -9999
    one_d_mask = [1, nodata, 2, nodata, 3]
    target = 8
    mask = xr.DataArray(
        np.array([one_d_mask] * grid_size),
        dims=("y", "x"),
        name="mask",
        attrs={"nodata": nodata},
    )

    new_data = xr.DataArray(
        np.full((grid_size, grid_size), target),
        dims=("y", "x"),
        name="layer2",
        attrs={"nodata": nodata},
    )
    grid_component.set(new_data, mask=mask)

    result = grid_component._data["layer2"].values
    assert (
        result[result == target].size
        == (len(one_d_mask) - one_d_mask.count(nodata)) * grid_size
    )
    assert (result[result != target] == nodata).all()


def test_set_mask_by_name_from_data(grid_component: GridComponent):
    nodata = -9999
    one_d_mask = [nodata, 1, nodata, 1, nodata]
    grid_size = 5
    target = 9

    layer3 = xr.DataArray(
        np.full((grid_size, grid_size), target),
        dims=("y", "x"),
        name="layer3",
        attrs={"nodata": nodata},
    )
    mask = xr.DataArray(
        np.array([one_d_mask] * grid_size),
        dims=("y", "x"),
        name="mask",
        attrs={"nodata": nodata},
    )
    data = xr.Dataset({"layer3": layer3, "mask": mask}, attrs={"nodata": nodata})

    grid_component.set(data, mask="mask")
    result = grid_component._data["layer3"]
    assert result.values[result == target].size == one_d_mask.count(1) * grid_size
    assert (result.values[result != target] == nodata).all()


def test_set_mask_by_name_from_existing(grid_component: GridComponent):
    grid_size = 5
    nodata = -9999
    one_d_mask = [1, nodata, 1, nodata, 1]
    target = 10

    mask = xr.DataArray(
        np.array([one_d_mask] * grid_size),
        dims=("y", "x"),
        name="mask",
        attrs={"nodata": nodata},
    )
    grid_component.set(mask)

    new_data = xr.DataArray(
        np.full((grid_size, grid_size), target),
        dims=("y", "x"),
        name="layer4",
        attrs={"nodata": nodata},
    )

    grid_component.set(new_data, mask="mask")
    result = grid_component._data["layer4"]

    assert (
        result.values[result == target].size
        == (len(one_d_mask) - one_d_mask.count(nodata)) * grid_size
    )
    assert (result.values[result != target] == nodata).all()


def test_set_mask_not_found(grid_component: GridComponent):
    grid_size = 5
    target = 11

    new_data = xr.DataArray(
        np.full((grid_size, grid_size), target), dims=("y", "x"), name="layer5"
    )

    grid_component.set(new_data, mask="not_found")
    # Should not raise, just no masking
    assert (grid_component._data["layer5"] == target).all()


def test_boolean_layer_with_mask(grid_component: GridComponent):
    grid_size = 5
    nodata = -9999
    one_d_mask = [1, nodata, 1, nodata, 1]

    layer = xr.DataArray(
        np.ones((grid_size, grid_size), dtype=bool), dims=("y", "x"), name="bool_layer"
    )
    mask = xr.DataArray(
        np.array([one_d_mask] * grid_size),
        dims=("y", "x"),
        name="mask",
        attrs={"nodata": nodata},
    )

    grid_component.set(mask)
    grid_component.set(layer, mask="mask")

    result = grid_component._data["bool_layer"]
    assert result.dtype == bool
    assert result.values[result == False].size == one_d_mask.count(nodata) * grid_size


@pytest.mark.parametrize(
    ("lats", "force_sn", "should_flip"),
    [
        ([1.0, 2.0], True, False),  # data=SN, force_sn=True => no flip
        ([2.0, 1.0], True, True),  # data=NS, force_sn=True => flip
        ([1.0, 2.0], False, False),  # data=SN, force_sn=False => no flip
        ([2.0, 1.0], False, False),  # data=NS, force_sn=False => no flip
    ],
)
def test_set_with_flipud(
    lats: list[float],
    force_sn: bool,
    should_flip: bool,
    grid_component: GridComponent,
):
    original_data = np.array([[1, 2], [3, 4]])
    expected_data = np.flipud(original_data) if should_flip else original_data

    data_array = xr.DataArray(
        data=original_data,
        coords={"lat": lats, "lon": [10.0, 11.0]},
        dims=["lat", "lon"],
        name="test_var",
    )
    grid_component.set(data_array, force_sn=force_sn)

    assert (grid_component.data["test_var"].values == expected_data).all()
