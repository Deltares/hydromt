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
            "grid": {"type": "GridComponent"},
            "subgrid": {"type": "GridComponent"},
        },  # type: ignore
        model_type=Model,
    )
    assert globals.components[0].name == "grid"
    assert globals.components[0].type == GridComponent
    assert globals.components[1].name == "subgrid"
    assert globals.components[1].type == GridComponent


def test_validate_global_config_components_wrong_input():
    with pytest.raises(TypeError, match="'str' object is not a mapping"):
        HydromtGlobalConfig(components={"grid": "foo"}, model_type=Model)
    with pytest.raises(TypeError, match="'NoneType' object is not a mapping"):
        HydromtGlobalConfig(components={"grid": None}, model_type=Model)


def test_validate_global_config_model_type():
    class Foo:
        pass

    with pytest.raises(ValidationError, match="Input should be a subclass of Model"):
        HydromtGlobalConfig(components=[], model_type=Foo)
    HydromtGlobalConfig(components=[], model_type=Model)
    HydromtGlobalConfig(components=[], model_type="Model")
