import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hydromt.model.model import Model

from hydromt._utils.rgetattr import _rgetattr

__all__ = ["_validate_steps"]


def _validate_steps(model: "Model", steps: list[dict[str, dict[str, Any]]]) -> None:
    for step_dict in steps:
        step, options = next(iter(step_dict.items()))
        attr = _rgetattr(model, step, None)
        if attr is None:
            raise AttributeError(f"Method {step} not found in model.")
        if not hasattr(attr, "__ishydromtstep__"):
            raise AttributeError(
                f"Method {step} is not allowed to be called on model, since it is not a HydroMT step definition."
                " Add @hydromt_step if that is your intention."
            )

        # attribute found, validate keyword arguments
        # Throws if bind fails.
        sig = inspect.signature(attr)
        options = options or {}
        _ = sig.bind(**options)
