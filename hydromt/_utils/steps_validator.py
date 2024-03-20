import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hydromt.models.model import Model

from .rgetattr import rgetattr


def validate_steps(model: "Model", steps: list[dict[str, dict[str, Any]]]) -> None:
    for step in steps:
        step_name, options = next(iter(step.items()))
        attr = rgetattr(model, step_name, None)
        if attr is None:
            raise AttributeError(f"Method {step_name} not found in model.")
        if not getattr(attr, "__ishydromtstep__", False) == True:
            raise AttributeError(
                f"Method {step_name} is not allowed to be called on model, since it is not a HydroMT step definition."
                " Add @hydromt_step if that is your intention."
            )

        # attribute found, validate keyword arguments
        # Throws if bind fails.
        sig = inspect.signature(attr)
        options = options or {}
        _ = sig.bind(**options)
