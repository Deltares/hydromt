import pytest
from pydantic import ValidationError

from hydromt._validators.model_config import (
    ComponentSpec,
    ModelSpec,
    WorkflowStep,
)
from hydromt.model.components.base import ModelComponent
from hydromt.model.components.grid import (
    GridComponent,
)
from hydromt.model.model import Model
from hydromt.model.steps import hydromt_step


@pytest.fixture
def mock_component(mock_model: Model) -> GridComponent:
    return GridComponent(mock_model)


def test_write_validation_validation(mock_model: Model):
    d = {"components": ["config", "geoms", "grid"]}
    step = WorkflowStep(name="write", kwargs=d)
    step.bind_to_model(mock_model)
    assert step._method == mock_model.write


def test_validate_component_config_wrong_characters():
    with pytest.raises(
        ValidationError, match=r"1test is not a valid Python identifier"
    ):
        ComponentSpec(name="1test", type=ModelComponent)


def test_validate_component_config_reserved_keyword():
    with pytest.raises(ValidationError, match=r"import is a Python reserved keyword"):
        ComponentSpec(name="import", type=ModelComponent)


def test_validate_component_config_unknown_component():
    with pytest.raises(KeyError, match="FooComponent"):
        ComponentSpec(name="foo", type="FooComponent")


def test_validate_component_config_known_component():
    ComponentSpec(name="foo", type="GridComponent")
    ComponentSpec(name="foo", type=ModelComponent)


def test_validate_component_no_input():
    with pytest.raises(ValidationError, match="Field required"):
        ComponentSpec(type=ModelComponent)
    with pytest.raises(ValidationError, match="Field required"):
        ComponentSpec(name="foo")


def test_validate_step_signature_bind():
    class MyComponent:
        @hydromt_step
        def foo(self, a: int, b: str):
            pass

    class MyModel:
        bar: MyComponent
        components: dict[str, MyComponent]

        def __init__(self):
            self.bar = MyComponent()
            self.components = {"bar": self.bar}

        @hydromt_step
        def foobar(self, a: int, b: str):
            pass

    model = MyModel()

    s = WorkflowStep(name="foobar", kwargs={"b": "test"})
    with pytest.raises(ValueError, match="missing a required argument: 'a'"):
        s.bind_to_model(model)

    s = WorkflowStep(name="foobar", kwargs={"a": 1})
    with pytest.raises(ValueError, match="missing a required argument: 'b'"):
        s.bind_to_model(model)

    s = WorkflowStep(name="bar.foo", kwargs={"b": "test"})
    with pytest.raises(ValueError, match="missing a required argument: 'a'"):
        s.bind_to_model(model)

    s = WorkflowStep(name="bar.foo", kwargs={"a": 1})
    with pytest.raises(ValueError, match="missing a required argument: 'b'"):
        s.bind_to_model(model)

    s = WorkflowStep(name="foobar", kwargs={"a": 1, "b": "test"})
    s.bind_to_model(model)
    assert s._method == model.foobar


def test_validate_global_config_components():
    globals = ModelSpec(
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
        ModelSpec(components={"grid": "foo"})
    with pytest.raises(TypeError, match="'NoneType' object is not a mapping"):
        ModelSpec(components={"grid": None})
