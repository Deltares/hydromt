"""Provides the base class for model components."""

from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING, Dict, cast
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
    def _root(self) -> "ModelRoot":
        """Return the root of the model this component is associated with."""
        return self._model.root

    def test_equal(self, other: "ModelComponent") -> tuple[bool, Dict[str, str]]:
        """Test if two components are equal.

        Inherit this method in the subclass to test for equality on data.
        Don't forget to call super().test_equal(other) in the subclass.

        Parameters
        ----------
        other : ModelComponent
            The component to compare against.

        Returns
        -------
        tuple[bool, Dict[str, str]]
            True if the components are equal, and a dict with the associated errors per property checked.
        """
        errors: Dict[str, str] = {}
        if not isinstance(other, self.__class__):
            errors["__class__"] = f"other does not inherit from {self.__class__}."
        return len(errors) == 0, errors
