"""HydroMT Model API."""

from hydromt.model.model import Model
from hydromt.model.root import ModelRoot
from hydromt.model.steps import hydromt_step
from hydromt.model.example.example_model import ExampleModel

__all__ = ["Model", "ExampleModel", "ModelRoot", "hydromt_step"]


# define hydromt model entry points
# see also hydromt.model group in pyproject.toml
__hydromt_eps__ = ["Model", "ExampleModel"]
