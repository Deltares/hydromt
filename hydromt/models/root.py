"." "the new model root class." ""

from logging import Logger, getLogger
from os import PathLike
from os.path import isdir
from pathlib import Path

from hydromt._typing import ModeLike, ModelMode
from hydromt.io.path import make_path_abs_and_cross_platform

logger = getLogger(__name__)


class ModelRoot:
    """A class to handle model roots in a cross platform manner."""

    def __init__(
        self,
        path: PathLike,
        mode: ModeLike = "w",
        logger: Logger = logger,
    ):
        self.mode = mode
        self.path = path
        self.logger: Logger = logger

    @property
    def path(self) -> Path:
        """The path to the model root in question."""
        return self._path

    @path.setter
    def path(self, path: PathLike) -> Path:
        """Set the path of the root, adjusting for cross platform where necessary. Returns the path that was set."""
        self._path = make_path_abs_and_cross_platform(path)

        if self.is_reading_mode():
            self._check_root_exists()

        return self._path

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

    def __repr__(self):
        return f"ModelRoot(path={self._path}, mode={self._mode})"

    def _check_root_exists(self):
        # check directory
        if not isdir(self._path):
            raise IOError(f'model root not found at "{self._path}"')
