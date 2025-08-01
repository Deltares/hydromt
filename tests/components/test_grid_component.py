import logging
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from pytest_mock import MockerFixture

from hydromt.model.components.grid import (
    GridComponent,
)
from hydromt.model.root import ModelRoot


def test_set_dataset(mock_model, hydds):
    grid_component = GridComponent(model=mock_model)
    grid_component.root.is_reading_mode.return_value = False
    grid_component.set(data=hydds)
    assert len(grid_component.data) > 0
    assert isinstance(grid_component.data, xr.Dataset)


def test_set_dataarray(mock_model, hydds):
    grid_component = GridComponent(model=mock_model)
    data_array = hydds.to_array()
    grid_component.root.is_reading_mode.return_value = False
    grid_component.set(data=data_array, name="data_array")
    assert "data_array" in grid_component.data.data_vars.keys()
    assert len(grid_component.data.data_vars) == 1


def test_set_raise_errors(mock_model, hydds):
    grid_component = GridComponent(model=mock_model)
    grid_component.root.is_reading_mode.return_value = False
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


def test_write(
    mock_model, tmpdir, caplog: pytest.LogCaptureFixture, mocker: MockerFixture
):
    grid_component = GridComponent(model=mock_model)
    grid_component.root.is_reading_mode.return_value = False
    # Test skipping writing when no grid data has been set
    caplog.set_level(logging.WARNING)
    grid_component.write()
    assert "No grid data found, skip writing" in caplog.text
    # Test raise IOerror when model is in read only mode
    mock_model.root = ModelRoot(tmpdir, mode="r")
    grid_component = GridComponent(model=mock_model)
    mocker.patch.object(GridComponent, "data", ["test"])
    with pytest.raises(IOError, match="Model opened in read-only mode"):
        grid_component.write()


def test_read(tmpdir, mock_model, hydds, mocker: MockerFixture):
    # Test for raising IOError when model is in writing mode
    grid_component = GridComponent(model=mock_model)
    mock_model.root = ModelRoot(path=tmpdir, mode="w")
    with pytest.raises(IOError, match="Model opened in write-only mode"):
        grid_component.read()
    mock_model.root = ModelRoot(path=tmpdir, mode="r+")
    grid_component = GridComponent(model=mock_model)
    mocker.patch(
        "hydromt.model.components.grid._read_ncs", return_value={"grid": hydds}
    )
    grid_component.read()
    assert grid_component.data == hydds


def test_properties(caplog: pytest.LogCaptureFixture, demda, mock_model):
    grid_component = GridComponent(mock_model)
    grid_component.root.is_reading_mode.return_value = False
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


def test_initialize_grid(mock_model, tmpdir):
    mock_model.root = ModelRoot(path=tmpdir, mode="r")
    grid_component = GridComponent(mock_model)
    grid_component.read = MagicMock()
    grid_component._initialize_grid()
    assert isinstance(grid_component._data, xr.Dataset)
    assert grid_component.read.called
