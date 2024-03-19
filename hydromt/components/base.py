"""Provides the base class for model components."""

from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING, cast
from weakref import ReferenceType, ref

from hydromt.data_catalog import DataCatalog

if TYPE_CHECKING:
    from hydromt.models import Model
    from hydromt.models._root import ModelRoot


class ModelComponent(ABC):
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
    def model_root(self) -> "ModelRoot":
        """Return the root of the model this component is associated with."""
        return self.model.root

    @abstractmethod
    def read(self):
        """Read the file(s) into the component."""
        if not self.model_root.is_reading_mode():
            raise IOError("Model opened in write-only mode")

    @abstractmethod
    def write(self):
        """Write the component to file(s)."""
        if not self.model_root.is_writing_mode():
            raise IOError("Model opened in read-only mode")
