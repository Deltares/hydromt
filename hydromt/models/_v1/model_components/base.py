"""the new model component class"""

from typing import Any, Optional
from weakref import ref
from ..model import Model


class ModelComponent:
    def __init__(self, model: Model):
        self._data: Optional[Any] = None
        self._model_ref: ref[Model] = ref(model)

    @property
    def model(self) -> ref[Model]:
        return self._model_ref

    @property
    def data(self) -> Optional[Any]:
        return self._data

    def set_data(self, value: Any) -> None:
        self._data = value
