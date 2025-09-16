"""the new model root class."""

import logging
from os.path import isdir
from pathlib import Path
from shutil import SameFileError, copyfile
from typing import Optional

from hydromt._typing import ModeLike, ModelMode
from hydromt._utils.log import (
    _add_filehandler,
    _setuplog,
    get_hydromt_logger,
    remove_hydromt_file_handlers,
)

logger = get_hydromt_logger(__name__)

__all__ = ["ModelRoot"]


class ModelRoot:
    """A class to handle model roots in a cross platform manner."""

    def __init__(
        self,
        path: Path,
        mode: ModeLike = "w",
        log_file: Path | None = None,
        log_level: int = logging.INFO,
    ):
        self._log_file: Path | None = log_file
        self._log_level: int = log_level
        self.mode = mode

        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)

        _setuplog(
            path=self.path / log_file if log_file is not None else None,
            log_level=self._log_level,
            append=self.mode == ModelMode.APPEND,
        )

        self.set(path, mode, log_file)

    def set(
        self,
        path: Path,
        mode: Optional[ModeLike] = None,
        log_file: Optional[Path] = None,
        log_level: Optional[int] = None,
    ) -> Path:
        """Set the path of the root, and create the necessary loggers."""
        logger.info(f"Setting root to {path}")

        current_root = self.path
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        if mode:
            self.mode = mode

        if self.is_reading_mode():
            self._check_root_exists()

        # Decide new log file location
        _fn = log_file or self._log_file or None
        new_log_file = self.path / _fn if _fn is not None else None

        # Remove old log file handlers
        if self._log_file is not None:
            old_log_file = current_root / self._log_file
            remove_hydromt_file_handlers(path_or_filename=old_log_file)

            # Copy old log file, will get cleared below if in overwrite mode
            if new_log_file is not None:
                new_log_file.parent.mkdir(parents=True, exist_ok=True)
                if old_log_file.exists():
                    try:
                        copyfile(old_log_file, new_log_file)
                    except SameFileError:
                        pass

        # Add new log file handler
        if new_log_file is not None:
            self._log_level = log_level or self._log_level
            # Clear logs if in overwrite mode
            if self.mode == ModelMode.FORCED_WRITE:
                new_log_file.unlink(missing_ok=True)
            _add_filehandler(path=new_log_file, log_level=self._log_level)
            self._log_file = new_log_file.relative_to(self.path)
        else:
            self._log_file = None

        return self.path

    def close(self) -> None:
        """Close the model root, removing any open log file handlers."""
        if self._log_file is not None:
            remove_hydromt_file_handlers(path_or_filename=self.path / self._log_file)
            self._log_file = None

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

    def _check_root_exists(self):
        # check directory
        if not isdir(self.path):
            raise IOError(f'model root not found at "{self.path}"')
