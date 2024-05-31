"""HydroMT Model API."""

from .hydromt_step import hydromt_step
from .model import Model

__all__ = ["Model", "hydromt_step"]

# define hydromt model entry points
# see also hydromt.model group in pyproject.toml
__hydromt_eps__ = ["Model"]
