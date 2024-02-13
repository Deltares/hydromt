import logging
from os.path import join
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from hydromt.data_catalog import DataCatalog
from hydromt.models.components.grid import GridComponent
from hydromt.models.root import ModelRoot

logger = logging.getLogger(__name__)
logger.propagate = True


def test_set(hydds, tmp_dir, rioda):
    model_root = ModelRoot(path=tmp_dir)
    data_catalog = DataCatalog()
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None
    )
    # Test setting xr.Dataset
    grid_component.set(data=hydds)
    assert len(grid_component._data) > 0
    assert isinstance(grid_component._data, xr.Dataset)
    # Test setting xr.DataArray
    data_array = hydds.to_array()
    grid_component.set(data=data_array, name="data_array")
    assert "data_array" in grid_component._data.data_vars.keys()
    assert len(grid_component._data.data_vars) == 3
    # Test setting nameless data array
    data_array.name = None
    with pytest.raises(
        ValueError,
        match=f"Unable to set {type(data_array).__name__} data without a name",
    ):
        grid_component.set(data=data_array)
    # Test setting np.ndarray of different shape
    ndarray = np.random.rand(4, 5)
    with pytest.raises(ValueError, match="Shape of data and grid maps do not match"):
        grid_component.set(ndarray, name="ndarray")


def test_write(tmp_dir, caplog):
    model_root = ModelRoot(path=tmp_dir)
    data_catalog = DataCatalog()
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None, logger=logger
    )
    # Test skipping writing when no grid data has been set
    caplog.set_level(logging.WARNING)
    grid_component.write()
    assert "No grid data found, skip writing" in caplog.text
    # Test raise IOerror when model is in read only mode
    model_root = ModelRoot(tmp_dir, mode="r")
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None
    )
    with patch.object(GridComponent, "_data", ["test"]):
        with pytest.raises(IOError, match="Model opened in read-only mode"):
            grid_component.write()


def test_read(tmp_dir, hydds):
    # Test for raising IOError when model is in writing mode
    model_root = ModelRoot(path=tmp_dir, mode="w")
    data_catalog = DataCatalog()
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None, logger=logger
    )
    with pytest.raises(IOError, match="Model opened in write-only mode"):
        grid_component.read()
    model_root = ModelRoot(path=tmp_dir, mode="r+")
    data_catalog = DataCatalog()
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None, logger=logger
    )
    with patch("hydromt.models.components.grid.read_nc", return_value={"grid": hydds}):
        grid_component.read()
        assert grid_component._data == hydds


def test_create(tmp_dir):
    model_root = ModelRoot(path=join(tmp_dir, "grid_model"))
    data_catalog = DataCatalog(data_libs=["artifact_data"])
    grid_component = GridComponent(
        root=model_root, data_catalog=data_catalog, model_region=None
    )
    # Wrong region kind
    with pytest.raises(ValueError, match="Region for grid must be of kind"):
        grid_component.create(region={"vector_model": "test_model"})
    # bbox
    bbox = [12.05, 45.30, 12.85, 45.65]
    with pytest.raises(
        ValueError, match="res argument required for kind 'bbox', 'geom'"
    ):
        grid_component.create({"bbox": bbox})
    grid_component.create(
        region={"bbox": bbox},
        res=0.05,
        add_mask=False,
        align=True,
    )

    assert "mask" not in grid_component._data
    # assert model.crs.to_epsg() == 4326
    # assert model.grid.raster.dims == ("y", "x")
    # assert model.grid.raster.shape == (7, 16)
    # assert np.all(np.round(model.grid.raster.bounds, 2) == bbox)
    # grid = model.grid
    # model._grid = xr.Dataset()  # remove old grid


def test_properties():
    pass


def test_initialize_grid():
    pass


def test_set_crs():
    pass


def test_add_data_from_constant():
    pass


def test_add_data_from_rasterdataset():
    pass


def test_add_data_from_raster_reclass():
    pass


def test_add_data_from_geodataframe():
    pass
