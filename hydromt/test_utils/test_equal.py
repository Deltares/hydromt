"""Test utilities for comparing model components."""

from typing import Tuple

import geopandas as gpd
import xarray as xr
from geopandas.testing import assert_geodataframe_equal

from hydromt.components.base import ModelComponent
from hydromt.models.model import Model


def check_models_equal(one: Model, other: Model) -> Tuple[bool, dict[str, str]]:
    """Test if two models including their data components are equal.

    Parameters
    ----------
    other : Model (or subclass)
        Model to compare against

    Returns
    -------
    equal: bool
        True if equal
    errors: dict
        Dictionary with errors per model component which is not equal
    """
    assert isinstance(other, type(one))
    components = list(one._components.keys())
    components_other = list(other._components.keys())
    assert components == components_other
    errors: dict[str, str] = {}
    is_equal = True
    for c in components:
        component_equal, component_errors = check_components_equal(
            getattr(one, c), getattr(other, c), c
        )
        is_equal &= component_equal
        errors.update(**component_errors)
    return is_equal, errors


def check_components_equal(
    one: ModelComponent, other: ModelComponent, property_name: str = ""
) -> Tuple[bool, dict[str, str]]:
    """Recursive test of model components.

    Returns if the components are equal, and a dict with component name and associated error message.
    """
    errors: dict[str, str] = {}
    try:
        assert isinstance(other, type(one)), "property types do not match"
        if hasattr(one, "data"):
            assert hasattr(other, "data")
            property_name = f"{property_name}.data"
            if isinstance(one.data, xr.Dataset):
                assert isinstance(other.data, xr.Dataset)
                xr.testing.assert_allclose(one.data, other.data)
            elif isinstance(one.data, gpd.GeoDataFrame):
                assert_geodataframe_equal(one.data, other.data)
    except AssertionError as e:
        errors.update({property_name: str(e)})
    return len(errors) == 0, errors
