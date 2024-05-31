"""HydroMT Model API."""

from .model import Model

__all__ = ["Model"]

# define hydromt model entry points
# see also hydromt.model group in pyproject.toml
__hydromt_eps__ = ["Model"]
