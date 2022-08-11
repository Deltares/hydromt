# -*- coding: utf-8 -*-
"""Tests for the hydromt.models module of HydroMT"""

import pytest
import xarray as xr
import numpy as np
from hydromt.models.model_api import _check_data
from hydromt.models import Model, GridModel


def test_check_data(demda):
    data_dict = _check_data(demda.copy(), "elevtn")
    assert isinstance(data_dict["elevtn"], xr.DataArray)
    assert data_dict["elevtn"].name == "elevtn"
    with pytest.raises(ValueError, match="Name required for DataArray"):
        _check_data(demda)
    demda.name = "dem"
    demds = demda.to_dataset()
    data_dict = _check_data(demds, "elevtn", False)
    assert isinstance(data_dict["elevtn"], xr.Dataset)
    data_dict = _check_data(demds, split_dataset=True)
    assert isinstance(data_dict["dem"], xr.DataArray)
    with pytest.raises(ValueError, match="Name required for Dataset"):
        _check_data(demds, split_dataset=False)
    with pytest.raises(ValueError, match='Data type "dict" not recognized'):
        _check_data({"wrong": "type"})


def test_model(model, tmpdir):
    non_compliant = model._test_model_api()
    # Staticmaps -> moved from _test_model_api as it is deprecated
    if not isinstance(model.staticmaps, xr.Dataset):
        non_compliant.append("staticmaps")
    assert len(non_compliant) == 0, non_compliant
    # write model
    model.set_root(str(tmpdir), mode="w")
    model.write()
    # read model
    model1 = Model(str(tmpdir), mode="r")
    model1.read()
    # check if equal
    model._results = {}  # reset results for comparison
    equal, errors, components = model._test_equal(model1)
    assert equal, errors
    comp = [
        "config",
        "crs",
        "forcing",
        "geoms",
        "maps",
        "region",
        "results",
        "states",
        "staticgeoms",
        "staticmaps",
    ]
    assert np.all([c in components for c in comp])


def test_gridmodel(grid_model, tmpdir):
    non_compliant = grid_model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # grid specific attributes
    assert np.all(grid_model.res == grid_model.grid.raster.res)
    assert np.all(grid_model.bounds == grid_model.grid.raster.bounds)
    assert np.all(grid_model.transform == grid_model.grid.raster.transform)
    # write model
    grid_model.set_root(str(tmpdir), mode="w")
    grid_model.write()
    # read model
    model1 = GridModel(str(tmpdir), mode="r")
    model1.read()
    # check if equal
    equal, errors, components = grid_model._test_equal(model1)
    assert equal, errors
    assert np.all([c in components for c in ["grid"]])
