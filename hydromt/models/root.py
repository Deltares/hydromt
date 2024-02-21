"""the new model root class."""

from logging import FileHandler, Logger, getLogger
from os import mkdir, remove
from os.path import dirname, exists, isdir, join
from pathlib import Path
from shutil import copyfile
from typing import Optional

from hydromt._typing import ModeLike, ModelMode
from hydromt._typing.type_def import StrPath
from hydromt._utils.log import add_filehandler

logger = getLogger(__name__)


class ModelRoot:
    """A class to handle model roots in a cross platform manner."""

    def __init__(
        self,
        path: StrPath,
        mode: ModeLike = "w",
        logger: Logger = logger,
    ):
        self.set(path, mode, logger)

    @property
    def path(self) -> Path:
        """The path to the model root in question."""
        return self._path

    def set(
        self,
        path: StrPath,
        mode: Optional[ModeLike] = None,
        logger: Logger = logger,
    ) -> Path:
        """Set the path of the root, and create the necessary loggers."""
        if hasattr(self, "_path"):
            old_path = self._path
        else:
            old_path = None

        self.logger = logger
        self.logger.info(f"setting root to {path}")

        self._path = Path(path)

        if mode:
            self._mode = ModelMode.from_str_or_mode(mode)

        if self.is_reading_mode():
            self._check_root_exists()

        if self.mode is not None:
            is_override = self.mode.is_override()
        else:
            is_override = False

        self._create_loggers(old_path, is_override)
        return self._path

    def _close_logs(self):
        for _ in range(len(self.logger.handlers)):
            l = self.logger.handlers.pop()
            l.flush()
            l.close()

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

    # old_root is so we can copy over the old log file if needed.
    def _create_loggers(self, old_path: Optional[Path] = None, overwrite: bool = False):
        new_path = join(self._path, "hydromt.log")
        if old_path == new_path:
            return

        os.makedirs(self._path, exist_ok=True)

        log_level = 20  # default, but overwritten by the level of active loggers
        for i, h in enumerate(self.logger.handlers):
            log_level = h.level
            if isinstance(h, FileHandler):
                if dirname(h.baseFilename) != self._path:
                    self.logger.handlers.pop(i).close()
                break

        if overwrite and exists(new_path):
            remove(new_path)

        if old_path is not None and exists(old_path):
            old_log_path = join(old_path, "hydromt.log")
            if exists(old_log_path):
                copyfile(old_log_path, new_path)

        add_filehandler(self.logger, new_path, log_level)

    def __repr__(self):
        return f"ModelRoot(path={self._path}, mode={self._mode})"

    def _check_root_exists(self):
        # check directory
        if not isdir(self._path):
            raise IOError(f'model root not found at "{self._path}"')
