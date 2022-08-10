# -*- coding: utf-8 -*-
"""Tests for the hydromt.models module of HydroMT"""

import pytest
import xarray as xr
from hydromt.models.model_api import _check_data
from hydromt.models import Model


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
    root = str(tmpdir.join("model"))
    model.set_root(root, mode="w")
    model.write()
    # read model
    mod1 = Model(root, mode="r")
    mod1.read()
    # TODO check if identical


def test_gridmodel(grid_model):
    non_compliant = grid_model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # grid specific attributes
