"""Pydantic models for the validation of model config files."""
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Type

from pydantic import BaseModel, ConfigDict, Field

from hydromt.models import Model


class HydromtModelStep(BaseModel):
    """A Pydantic model for the validation of model setup functions."""

    model: Type[Model]
    fn: Callable
    args: Dict[str, Any] = Field(default_factory=dict)

    model_config: ConfigDict = ConfigDict(
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @staticmethod
    def from_dict(input_dict: Dict[str, Any], model: Model):
        """Generate a validated model of a step in a model config files."""
        fn_name, arg_dict = next(iter(input_dict.items()))
        fn_name = fn_name.strip("0123456789")
        try:
            fn = getattr(model, fn_name)
        except AttributeError:
            raise ValueError(
                f"Model of type {model.__name__} does not have function {fn_name}"
            )

        sig = signature(fn)
        sig_has_var_keyword = (
            len(
                [
                    param_name
                    for param_name, param in sig.parameters.items()
                    if param.kind == Parameter.VAR_KEYWORD
                ]
            )
            > 0
        )  # I think there can only be one of these
        unknown_parameters = {
            param_name
            for param_name, param_value in arg_dict.items()
            if param_name not in sig.parameters.keys()
            and (not isinstance(param_value, dict) and not sig_has_var_keyword)
        }

        if len(unknown_parameters) > 0:
            raise ValueError(
                f"Unknown parameters for function {fn_name}:{unknown_parameters}"
            )

        return HydromtModelStep(model=model, fn=fn, args=arg_dict)


class HydromtModelSetup(BaseModel):
    """A Pydantic model for the validation of model setup files."""

    steps: List[HydromtModelStep]

    @staticmethod
    def from_dict(input_dict: Dict[str, Any], model: Model):
        """Generate a validated model of a sequence steps in a model config file."""
        return HydromtModelSetup(
            steps=[
                HydromtModelStep.from_dict({fn_name: fn_args}, model)
                for fn_name, fn_args in input_dict.items()
            ]
        )
