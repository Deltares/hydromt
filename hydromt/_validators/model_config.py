"""Pydantic models for the validation of model config files."""

import inspect
from keyword import iskeyword
from typing import Any, Callable, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hydromt.components.base import ModelComponent
from hydromt.models import Model
from hydromt.plugins import PLUGINS


class HydromtComponentConfig(BaseModel):
    name: str
    type: Type[ModelComponent]

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        if not name.isidentifier():
            raise ValueError(f"{name} is not a valid python identifier")
        if iskeyword(name):
            raise ValueError(f"{name} is a python reserved keyword")
        return name

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v: str) -> Type[ModelComponent]:
        return PLUGINS.component_plugins[v]

    @model_validator(mode="before")
    @classmethod
    def transform_config(cls, v: dict[str, Any]):
        # The components list is a list of dictionaries with a single key-value pair.
        # The key is actually the name of the component.
        # The rest of the dictionary is the configuration for the component.
        name, rest = next(iter(v.items()))
        return {"name": name, **rest}

    model_config = ConfigDict(extra="forbid")


def _is_hydromt_step_function(attr: Any, function_name: str) -> bool:
    return (
        inspect.isfunction(attr)
        and attr.__name__ == function_name
        and attr.__ishydromtstep__
    )


class HydromtModelStep(BaseModel):
    fn: Callable
    args: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @staticmethod
    def from_dict(
        step_config: dict[str, dict[str, Any]],
        model_type: Type[Model],
        components: list[HydromtComponentConfig],
    ):
        """Generate a validated model of a step in a model config files."""
        name, options = next(iter(step_config.items()))
        split_name = name.split(".")
        if len(split_name) > 2:
            raise ValueError(
                f"Invalid step name {name}. Must be in the format <component>.<function> or <function> if the function is in the model itself."
            )
        function_owner: Type = model_type
        if len(split_name) == 2:
            component_config = next(x for x in components if x.name == split_name[0])
            function_owner = component_config.type

        members = inspect.getmembers(
            function_owner,
            predicate=lambda v: _is_hydromt_step_function(v, split_name[-1]),
        )

        if len(members) == 0:
            raise ValueError(
                f"Function {split_name[-1]} not found in {function_owner.__name__}"
            )

        # TODO: Test all arguments.

        return HydromtModelStep(fn=members[0][1], args=options)


class HydromtGlobalConfig(BaseModel):
    model_type: type[Model]
    components: list[HydromtComponentConfig]

    @field_validator("model_type", mode="before")
    @classmethod
    def validate_model_type(cls, v: str) -> Type[Model]:
        return PLUGINS.model_plugins[v]

    model_config = ConfigDict(extra="forbid")


class HydromtModelSetup(BaseModel):
    """A Pydantic model for the validation of model setup files."""

    steps: list[HydromtModelStep]
    # TODO: Add alias for `global`
    globals: HydromtGlobalConfig

    # TODO: Change to a model validator, no need for `from_dict functions`
    @staticmethod
    def from_dict(input_dict: dict[str, Any]):
        """Generate a validated model of a sequence steps in a model config file."""
        global_config = HydromtGlobalConfig(**input_dict["global"])
        return HydromtModelSetup(
            globals=global_config,
            steps=[
                HydromtModelStep.from_dict(
                    step,
                    global_config.model_type,
                    global_config.components,
                )
                for step in input_dict["steps"]
            ],
        )
