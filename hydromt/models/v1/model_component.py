"""Abstract base class for ModelComponent."""

from abc import ABC, abstractmethod


class ModelComponent(ABC):
    """Abstract base class for ModelComponent."""

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
