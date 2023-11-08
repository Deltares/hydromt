"""Pydantic models for the validation of model config files."""
from typing import Any, Dict

from pydantic import BaseModel


class HydromtStep(BaseModel):
    """A Pydantic model for the validation of model config files."""

    fn: str
    args: Dict[str, Any]

    @staticmethod
    def from_dict(input_dict):
        """Generate a validated model of a step in a model config files."""
        fn_name, arg_dict = next(iter(input_dict.items()))
        # TODO figure out how to actually load the correct functions form the correct
        # namespace from the provided names
        return HydromtStep(fn=fn_name, args=arg_dict)
