import pytest
from pydantic import ValidationError

from hydromt._validators.model_config import (
    HydromtComponentConfig,
    HydromtGlobalConfig,
    HydromtModelStep,
)
from hydromt.components.base import ModelComponent
from hydromt.components.grid import GridComponent
from hydromt.models.model import Model


def test_setup_grid_from_constant_validation():
    d = {
        "constant": 0.01,
        "name": "c1",
        "dtype": "float32",
        "nodata": -99.0,
    }
    HydromtModelStep(fn=GridComponent.add_data_from_constant, args=d)


def test_setup_grid_from_rasterdataset_validation():
    d = {
        "raster_fn": "merit_hydro_1k",
        "variables": ["elevtn", "basins"],
        "reproject_method": ["average", "mode"],
    }
    HydromtModelStep(fn=GridComponent.add_data_from_rasterdataset, args=d)


def test_setup_grid_from_geodataframe_validation():
    d = {
        "vector_fn": "hydro_lakes",
        "variables": ["waterbody_id", "Depth_avg"],
        "nodata": [-1, -999.0],
        "rasterize_method": "value",
        "rename": {"waterbody_id": "lake_id", "Detph_avg": "lake_depth"},
    }
    HydromtModelStep(fn=GridComponent.add_data_from_geodataframe, args=d)


def test_setup_grid_from_raster_reclass_validation():
    d = {
        "raster_fn": "vito",
        "reclass_table_fn": "vito_reclass",
        "reclass_variables": ["manning"],
        "reproject_method": ["average"],
    }
    HydromtModelStep(fn=GridComponent.add_data_from_raster_reclass, args=d)


def test_write_validation_validation():
    d = {"components": ["config", "geoms", "grid"]}
    HydromtModelStep(fn=Model.write, args=d)


def test_validate_component_config_wrong_characters():
    with pytest.raises(ValueError, match="is not a valid python identifier"):
        HydromtComponentConfig(name="1test", type=ModelComponent)


def test_validate_component_config_reserved_keyword():
    with pytest.raises(ValueError, match="is a python reserved keyword"):
        HydromtComponentConfig(name="import", type=ModelComponent)


def test_validate_component_config_unknown_component():
    with pytest.raises(KeyError, match="FooComponent"):
        HydromtComponentConfig(name="foo", type="FooComponent")


def test_validate_component_config_known_component():
    HydromtComponentConfig(name="foo", type="ModelComponent")
    HydromtComponentConfig(name="foo", type=ModelComponent)


def test_validate_component_no_input():
    with pytest.raises(ValidationError, match="Field required"):
        HydromtComponentConfig(type=ModelComponent)
    with pytest.raises(ValidationError, match="Field required"):
        HydromtComponentConfig(name="foo")


def test_validate_step_signature_bind():
    def foo(a: int, b: str):
        pass

    class Bar:
        def foo(self, a: int, b: str):
            pass

    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        HydromtModelStep(fn=foo, args={"b": "test"})
    with pytest.raises(TypeError, match="missing a required argument: 'b'"):
        HydromtModelStep(fn=foo, args={"a": 1})
    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        HydromtModelStep(fn=Bar.foo, args={"b": "test"})
    with pytest.raises(TypeError, match="missing a required argument: 'b'"):
        HydromtModelStep(fn=Bar.foo, args={"a": 1})
    HydromtModelStep(fn=foo, args={"a": 1, "b": "test"})


def test_validate_global_config_components():
    globals = HydromtGlobalConfig(
        components={
            "grid": {"type": GridComponent.__name__},
            "subgrid": {"type": GridComponent.__name__},
        },  # type: ignore
    )
    assert globals.components[0].name == "grid"
    assert globals.components[0].type == GridComponent
    assert globals.components[1].name == "subgrid"
    assert globals.components[1].type == GridComponent


def test_validate_global_config_components_wrong_input():
    with pytest.raises(TypeError, match="'str' object is not a mapping"):
        HydromtGlobalConfig(components={"grid": "foo"})
    with pytest.raises(TypeError, match="'NoneType' object is not a mapping"):
        HydromtGlobalConfig(components={"grid": None})
