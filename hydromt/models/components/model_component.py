"""Abstract base class for ModelComponent."""
from abc import ABC, abstractmethod
from logging import Logger

from hydromt.data_catalog import DataCatalog
from hydromt.models.root import ModelRoot


class ModelComponent(ABC):
    """Abstract base class for ModelComponent."""

    @abstractmethod
    def __init__(
        self,
        root: ModelRoot,
        data_catalog: DataCatalog,
        model_region,  # TODO: add ModelRegion object when finished
        logger: Logger,
    ):
        self.logger = logger
        self.root = root
        self.data_catalog = data_catalog
        self.model_region = model_region

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
