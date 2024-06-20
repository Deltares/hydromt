import pytest

from hydromt._utils.steps_validator import _validate_steps
from hydromt.model import hydromt_step
from hydromt.model.components.base import ModelComponent
from hydromt.model.model import Model


class FooComponent(ModelComponent):
    @hydromt_step
    def create(self, a: int, b: str) -> None:
        pass

    @hydromt_step
    def with_defaults(self, a: int, b: str = "2") -> None:
        pass

    def read(self) -> None:
        pass

    def write(self) -> None:
        pass


class FooModel(Model):
    @hydromt_step
    def foo(self, a: int, b: str) -> None:
        pass

    @hydromt_step
    def bar(self) -> None:
        pass

    def baz(self) -> None:
        pass


def test_validate_steps_unknown_args_in_dict():
    steps = [{"foo.create": {"a": 1, "b": "2", "c": 3}}]
    model = Model()
    model.add_component("foo", FooComponent(model))
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'c'"):
        _validate_steps(model, steps)


def test_validate_steps_not_all_args_in_dict():
    steps = [{"foo.create": {"a": 1}}]
    model = Model()
    model.add_component("foo", FooComponent(model))
    with pytest.raises(TypeError, match="missing a required argument: 'b'"):
        _validate_steps(model, steps)


def test_validate_steps_correct():
    steps = [{"foo.create": {"a": 1, "b": "2"}}]
    model = Model()
    model.add_component("foo", FooComponent(model))
    _validate_steps(model, steps)


def test_validate_steps_in_model_correct(PLUGINS):
    # modify plugin for testing
    _ = PLUGINS.model_plugins
    PLUGINS._model_plugins._plugins["FooModel"] = {  # type: ignore
        "name": "foo",
        "type": FooModel,
        "plugin_name": "testing",
        "version": "999",
    }
    model = FooModel()
    _validate_steps(model, [{"foo": {"a": 1, "b": "2"}}, {"bar": None}])


def test_validate_steps_disallowed_function(PLUGINS):
    # modify plugin for testing
    _ = PLUGINS.model_plugins
    PLUGINS._model_plugins._plugins["FooModel"] = {  # type: ignore
        "name": "foo",
        "type": FooModel,
        "plugin_name": "testing",
        "version": "999",
    }
    model = FooModel()
    with pytest.raises(
        AttributeError,
        match="Method baz is not allowed to be called on model, since it is not a HydroMT step definition. Add @hydromt_step if that is your intention.",
    ):
        _validate_steps(model, [{"baz": None}])


def test_validate_steps_blacklisted_function():
    model = Model()
    model.add_component("foo", FooComponent(model))
    with pytest.raises(
        AttributeError, match="Method build is not allowed to be called on model."
    ):
        _validate_steps(model, [{"build": None}])


def test_validate_steps_correct_with_defaults():
    model = Model()
    model.add_component("foo", FooComponent(model))
    _validate_steps(model, [{"foo.with_defaults": {"a": 1}}])
    _validate_steps(model, [{"foo.with_defaults": {"a": 1, "b": "3"}}])
