# -*- coding: utf-8 -*-
"""Tests for the hydromt.models module of HydroMT"""

import pytest
import xarray as xr
from hydromt.models.model_api import _check_data


def test_check_data(demda):
    data_dict = _check_data(demda.copy(), "elevtn")
    assert isinstance(data_dict["elevtn"], xr.DataArray)
    assert data_dict["elevtn"].name == "elevtn"
    with pytest.raises(ValueError, match="Name required for DataArray"):
        _check_data(demda)
    demda.name = "dem"
    demds = demda.to_dataset()
    data_dict = _check_data(demds, "elevtn")
    assert isinstance(data_dict["elevtn"], xr.Dataset)
    data_dict = _check_data(demds, None, True)
    assert isinstance(data_dict["dem"], xr.DataArray)
    with pytest.raises(ValueError, match="Name required for Dataset"):
        _check_data(demds)
    with pytest.raises(ValueError, match='Data type "dict" not recognized'):
        _check_data({"wrong": "type"})


def test_model(model):
    non_compliant = model._test_model_api()
    # Staticmaps -> moved from test as it is deprecated
    if not isinstance(model.staticmaps, xr.Dataset):
        non_compliant.append("staticmaps")
    assert len(non_compliant) == 0, non_compliant


def test_gridmodel(grid_model):
    non_compliant = grid_model._test_model_api()
    assert len(non_compliant) == 0, non_compliant
    # grid specific attributes
