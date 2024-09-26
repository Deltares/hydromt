"""Pydantic models for the validation of model config files."""

import inspect
from keyword import iskeyword
from typing import Any, Callable, Type, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from hydromt.model import Model
from hydromt.model.components.base import ModelComponent
from hydromt.plugins import PLUGINS


class HydromtComponentConfig(BaseModel):
    name: str
    type: Type[ModelComponent]

    model_config = ConfigDict(extra="forbid")

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
    def validate_type(cls, v):
        if isinstance(v, str):
            return PLUGINS.component_plugins[v]
        return v


class HydromtModelStep(BaseModel):
    fn: Callable[..., Any]
    args: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="after")
    def validate_model(self):
        sig = inspect.signature(self.fn)
        if sig.parameters.get("self", None) is not None:
            _ = sig.bind(**{"self": None, **self.args})
        else:
            _ = sig.bind(**self.args)

        return self


class HydromtGlobalConfig(BaseModel):
    components: list[HydromtComponentConfig]

    model_config = ConfigDict(extra="forbid")

    @field_validator("components", mode="before")
    @classmethod
    def validate_components(cls, v):
        if isinstance(v, dict):
            return [{"name": name, **options} for name, options in v.items()]
        return v


class HydromtModelSetup(BaseModel):
    """A Pydantic model for the validation of model setup files."""

    modeltype: Type[Model]
    steps: list[HydromtModelStep]
    globals: HydromtGlobalConfig = Field(serialization_alias="global")

    @field_validator("modeltype", mode="before")
    @classmethod
    def validate_model_type(cls, v):
        if isinstance(v, str):
            return PLUGINS.model_plugins[v]
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, v):
        if isinstance(v, dict):
            global_config = HydromtGlobalConfig(**v["global"])
            return {
                "modeltype": v["modeltype"],
                "globals": global_config,
                "steps": [
                    cls._create_step(
                        modeltype=cls.validate_model_type(v["modeltype"]),
                        components=global_config.components,
                        step=step,
                    )
                    for step in v["steps"]
                ],
            }
        return v

    @staticmethod
    def _create_step(
        *,
        modeltype: Type[Model],
        components: list[HydromtComponentConfig],
        step: dict[str, dict[str, Any]],
    ) -> HydromtModelStep:
        name, options = next(iter(step.items()))
        split_name = name.split(".")
        if len(split_name) > 2:
            raise ValueError(
                f"Invalid step name {name}. Must be in the format <component>.<function> or <function> if the function is in the model itself."
            )

        function_owner: Type[Union[Model, ModelComponent]] = modeltype
        if len(split_name) == 2:
            function_owner = next(x.type for x in components if x.name == split_name[0])

        members = inspect.getmembers(
            function_owner,
            predicate=lambda v: HydromtModelSetup._is_hydromt_step_function(
                v, split_name[-1]
            ),
        )
        if len(members) == 0:
            raise ValueError(
                f"Function {split_name[-1]} not found in {function_owner.__name__} or not marked as @hydromt_step."
            )

        return HydromtModelStep(fn=members[0][1], args=options)

    @staticmethod
    def _is_hydromt_step_function(attr: Any, function_name: str) -> bool:
        return (
            inspect.isfunction(attr)
            and attr.__name__ == function_name
            and hasattr(attr, "__ishydromtstep__")
        )
