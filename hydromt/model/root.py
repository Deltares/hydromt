"""the new model root class."""

import logging
from pathlib import Path

from hydromt._utils.path import _check_directory
from hydromt.model.mode import ModelMode

logger = logging.getLogger(__name__)

__all__ = ["ModelRoot"]


class ModelRoot:
    """Handle model roots in a cross platform manner.

    Parameters
    ----------
    path : Path
        The path to the root of the model.
    mode : ModelMode | str, optional
        The mode of the model (e.g. 'r' for reading, 'w' for writing etc.),
        by default "w".
    """

    def __init__(
        self,
        path: Path,
        mode: ModelMode | str = "w",
    ):
        self._mode: ModelMode | None = None
        self._path: Path | None = None

        # Set the Root
        self.set(path=path, mode=mode)

    def __repr__(self):
        return f"ModelRoot(path={self.path}, mode={self._mode})"

    ## Private methods
    def _assert_write_mode(self) -> None:
        if not self.mode.is_writing_mode():
            raise IOError("Model opened in read-only mode")

    def _assert_read_mode(self) -> None:
        if not self.mode.is_reading_mode():
            raise IOError("Model opened in write-only mode")

    ## Properties
    @property
    def mode(self) -> ModelMode:
        """The mode of the model this object belongs to."""
        return self._mode

    @mode.setter
    def mode(self, mode: ModelMode | str) -> ModelMode:
        """Set the mode of the model."""
        self._mode: ModelMode = ModelMode.from_str_or_mode(mode)
        return self._mode

    @property
    def path(self) -> Path:
        """Return the path of the model root."""
        return self._path

    @path.setter
    def path(self, path: Path | str):
        """Set the path of the model."""
        path = Path(path).resolve()
        if self.is_reading_mode():
            _check_directory(path=path, fail=True)
        if self.is_writing_mode():
            _check_directory(path=path, fail=False)
        self._path = path

    ## Checks
    def is_reading_mode(self) -> bool:
        """Test whether we are in reading mode or not."""
        return self._mode.is_reading_mode()

    def is_writing_mode(self) -> bool:
        """Test whether we are in writing mode or not."""
        return self._mode.is_writing_mode()

    def is_override_mode(self) -> bool:
        """Test whether we are in override mode or not."""
        return self._mode.is_override_mode()

    ## Mutating methods
    def set(
        self,
        path: Path,
        mode: ModelMode | str | None = None,
    ) -> Path:
        """Set a new model root and mode.

        Parameters
        ----------
        path : Path
            The path to the root of the model.
        mode : ModelMode | str | None, optional
            The mode of the model. If not provided, the mode currently defined in the
            `ModelRoot` is used. By default None.

        Returns
        -------
        Path
            The path to the new model root.
        """
        if mode is not None:
            self.mode = mode
        self.path = path

        return self.path
