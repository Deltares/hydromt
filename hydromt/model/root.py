"""the new model root class."""

import logging
from pathlib import Path
from typing import Optional

from hydromt.typing import ModeLike, ModelMode

logger = logging.getLogger(__name__)

__all__ = ["ModelRoot"]


class ModelRoot:
    """A class to handle model roots in a cross platform manner."""

    def __init__(
        self,
        path: Path,
        mode: ModeLike = "w",
    ):
        self._mode = ModelMode.from_str_or_mode(mode)
        if self.is_reading_mode():
            self._check_root_exists(path)
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)

    def set(
        self,
        path: Path,
        mode: Optional[ModeLike] = None,
    ) -> Path:
        """Set the path and mode of the root."""
        if mode:
            self.mode = mode

        if self.is_reading_mode():
            self._check_root_exists(path)

        self.path = Path(path).resolve()

        return self.path

    def _assert_write_mode(self) -> None:
        if not self.mode.is_writing_mode():
            raise IOError("Model opened in read-only mode")

    def _assert_read_mode(self) -> None:
        if not self.mode.is_reading_mode():
            raise IOError("Model opened in write-only mode")

    @property
    def mode(self) -> ModelMode:
        """The mode of the model this object belongs to."""
        return self._mode

    @mode.setter
    def mode(self, mode: ModeLike) -> ModelMode:
        """Set the mode of the model."""
        self._mode: ModelMode = ModelMode.from_str_or_mode(mode)
        return self._mode

    def is_writing_mode(self) -> bool:
        """Test whether we are in writing mode or not."""
        return self._mode.is_writing_mode()

    def is_reading_mode(self) -> bool:
        """Test whether we are in reading mode or not."""
        return self._mode.is_reading_mode()

    def is_override_mode(self) -> bool:
        """Test whether we are in override mode or not."""
        return self._mode.is_override_mode()

    def __repr__(self):
        return f"ModelRoot(path={self.path}, mode={self._mode})"

    def _check_root_exists(self, path: Optional[Path] = None) -> None:
        path = path or self.path
        if not path.exists():
            raise IOError(f'model root not found at "{path}"')
