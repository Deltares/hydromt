"""Provides the base class for model components."""

from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING, cast
from weakref import ReferenceType, ref

from hydromt.data_catalog import DataCatalog

if TYPE_CHECKING:
    from hydromt.models import Model
    from hydromt.root import ModelRoot


class ModelComponent(ABC):
    """Abstract base class for ModelComponent."""

    def __init__(self, model: "Model"):
        self.__model_ref: ReferenceType["Model"] = ref(model)

    @abstractmethod
    def read(self):
        """Read the file(s) into the component."""
        ...

    @abstractmethod
    def write(self):
        """Write the component to file(s)."""
        ...

    @property
    def _model(self) -> "Model":
        """Return the model object this component is associated with."""
        return cast("Model", self.__model_ref())

    @property
    def _data_catalog(self) -> DataCatalog:
        """Return the data catalog of the model this component is associated with."""
        return self._model.data_catalog

    @property
    def _logger(self) -> Logger:
        """Return the logger of the model this component is associated with."""
        return self._model.logger

    @property
    def _model_root(self) -> "ModelRoot":
        """Return the root of the model this component is associated with."""
        return self._model.root
