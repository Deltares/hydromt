"""the new model root class."""

import shutil
from pathlib import Path
from typing import Optional

from hydromt._typing import ModeLike, ModelMode
from hydromt._utils.log import (
    add_filehandler,
    get_hydromt_logger,
    initialize_logging,
    remove_filehandler,
)

logger = get_hydromt_logger(__name__)

__all__ = ["ModelRoot"]


class ModelRoot:
    """A class to handle model roots in a cross platform manner."""

    def __init__(
        self,
        path: Path,
        mode: ModeLike = "w",
    ):
        initialize_logging()
        self._filehandler = None
        self._mode = ModelMode.from_str_or_mode(mode)

        if self.is_reading_mode():
            self._check_root_exists(path)

        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self._add_filehandler(
            path=self.path / "hydromt.log", append=self.is_reading_mode()
        )

    def set(
        self,
        path: Path,
        mode: Optional[ModeLike] = None,
    ) -> Path:
        """Set the path of the root, and create the necessary loggers."""
        if mode:
            self.mode = mode

        if self.is_reading_mode():
            self._check_root_exists(path)

        if path != self.path:
            logger.info(f"Setting root to {path}")
            previous_log_file = self.path / "hydromt.log"
            new_log_file = path / "hydromt.log"
            self._remove_filehandler()
            path.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(previous_log_file, new_log_file)
            self._add_filehandler(path=new_log_file, append=self.is_reading_mode())
            self.path = path.resolve()

        return self.path

    def close(self) -> None:
        """Close the model root, removing any open log file handlers."""
        self._remove_filehandler()

    def _add_filehandler(self, path: Path, append: bool = True):
        if self._filehandler is not None:
            logger.warning(
                f"Trying to add a filehandler when one already exists writing to {self._filehandler.baseFilename}. "
                f"Removing existing handler first."
            )
            self._remove_filehandler()
        if not append:
            path.unlink(missing_ok=True)

        self._filehandler = add_filehandler(path=path)

    def _remove_filehandler(self) -> None:
        if self._filehandler is not None:
            remove_filehandler(path=Path(self._filehandler.baseFilename))
            self._filehandler = None

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
