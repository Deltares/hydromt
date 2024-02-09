"""Handeling for the mode a HydroMT Model can be in."""
from enum import Enum
from typing import Union


class ModelMode(Enum):
    """Modes that the model can be in."""

    READ = "r"
    WRITE = "w"
    FORCED_WRITE = "w+"
    APPEND = "r+"

    @staticmethod
    def from_str_or_mode(s: Union["ModelMode", str]) -> "ModelMode":
        """Construct a model mode from either a string or return provided if it's already a mode."""
        if isinstance(s, ModelMode):
            return s

        if s == "r":
            return ModelMode.READ
        elif s == "r+":
            return ModelMode.APPEND
        elif s == "w":
            return ModelMode.WRITE
        elif s == "w+":
            return ModelMode.FORCED_WRITE
        else:
            raise ValueError(f"Unknown mode: {s}, options are: r, r+, w, w+")

    def is_writing_mode(self):
        """Asster whether mode is writing or not."""
        return self in [ModelMode.WRITE, ModelMode.FORCED_WRITE, ModelMode.APPEND]

    def is_reading_mode(self):
        """Asster whether mode is reading or not."""
        return self in [ModelMode.READ, ModelMode.APPEND]

    def is_override(self):
        """Asster whether mode is able to overwrite or not."""
        return self in [ModelMode.FORCED_WRITE, ModelMode.APPEND]
