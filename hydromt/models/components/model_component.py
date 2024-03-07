"""Abstract base class for ModelComponent."""
import weakref
from abc import ABC, abstractmethod
from logging import Logger

from hydromt.data_catalog import DataCatalog
from hydromt.models.api import Model
from hydromt.models.root import ModelRoot


class ModelComponent(ABC):
    """Abstract base class for ModelComponent."""

    def __init__(self, model: Model):
        self._model_ref = weakref.ref(model)

    @abstractmethod
    def set(self):
        """Set data."""
        pass

    @abstractmethod
    def write(self):
        """Write data."""
        pass

    @abstractmethod
    def read(self):
        """Read data."""
        pass

    @property
    def model(self) -> Model:
        """Access the Model instance through the weak reference."""
        return self._model_ref()

    @property
    def data_catalog(self) -> DataCatalog:
        """Access the model DataCatalog."""
        return self.model.data_catalog

    @property
    def logger(self) -> Logger:
        """Access the model logger."""
        return self.model.logger

    @property
    def model_root(self) -> ModelRoot:
        """Access the model root."""
        return self.model.root
