"""Provides the base class for model components."""

from logging import Logger
from typing import TYPE_CHECKING, cast
from weakref import ReferenceType, ref

from hydromt.data_catalog import DataCatalog
from hydromt.models.root import ModelRoot

if TYPE_CHECKING:
    from hydromt.models import Model


class ModelComponent:
    """Abstract base class for ModelComponent."""

    def __init__(self, model: "Model"):
        self._model_ref: ReferenceType["Model"] = ref(model)

    @property
    def model(self) -> "Model":
        """Return the model object this component is associated with."""
        return cast("Model", self._model_ref())

    @property
    def data_catalog(self) -> DataCatalog:
        """Return the data catalog of the model this component is associated with."""
        return self.model.data_catalog

    @property
    def logger(self) -> Logger:
        """Return the logger of the model this component is associated with."""
        return self.model.logger

    @property
    def model_root(self) -> ModelRoot:
        """Return the root of the model this component is associated with."""
        return self.model.root
