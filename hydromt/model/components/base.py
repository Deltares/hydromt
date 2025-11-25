"""Provides the base class for model components."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, cast
from weakref import ReferenceType, ref

import xarray as xr

from hydromt.data_catalog import DataCatalog
from hydromt.typing.deferred_file_close import DeferredFileClose

if TYPE_CHECKING:
    from hydromt.model import Model
    from hydromt.model.root import ModelRoot

logger = logging.getLogger(__name__)


class ModelComponent(ABC):
    """Abstract base class for ModelComponent."""

    def __init__(self, model: "Model"):
        self.__model_ref: ReferenceType["Model"] = ref(model)
        self._open_datasets: list[xr.Dataset] = []
        self._deferred_file_close_handles: list[DeferredFileClose] = []

    @abstractmethod
    def read(self):
        """Read the file(s) into the component."""
        ...

    @abstractmethod
    def write(self) -> None:
        """Write the component to file(s)."""
        ...

    @property
    def model(self) -> "Model":
        """Return the model object this component is associated with."""
        return cast("Model", self.__model_ref())

    @property
    def name_in_model(self) -> str:
        """Find the name of the component in the parent model components."""
        for name, component in self.model.components.items():
            if component is self:
                return name
        raise ValueError("Component not found in model components.")

    @property
    def data_catalog(self) -> DataCatalog:
        """Return the data catalog of the model this component is associated with."""
        return self.model.data_catalog

    @property
    def root(self) -> "ModelRoot":
        """Return the root of the model this component is associated with."""
        return self.model.root

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

    def close(self) -> None:
        """Clean up all open datasets. Method to be called before finish_write."""
        for ds in self._open_datasets:
            ds.close()
        self._open_datasets.clear()

    def finish_write(self):
        """Finish the write functionality after cleanup was called for all components in the model.

        All DeferredFileClose objects can overwrite any lazy loaded files now.
        """
        for close_handle in self._deferred_file_close_handles:
            close_handle.close()
        self._deferred_file_close_handles.clear()
