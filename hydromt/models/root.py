"""the new model root class."""

from logging import Logger, getLogger
from os import PathLike, mkdir, remove, rename
from os.path import dirname, exists, isdir, join
from pathlib import Path
from typing import Optional
from data.src.era5_download_resample_convert import move_replace

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
        self.logger = logger
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
        self.logger.info(f"setting path to {path}")

        self._path = Path(path)

        if mode:
            self._mode = ModelMode.from_str_or_mode(mode)

        if self.is_reading_mode():
            self._check_root_exists()

        self._create_loggers(old_path, self.mode.is_override())
        return self._path

    def _close_logs(self):
        for _ in range(len(self.logger.handlers)):
            l = self.logger.handlers.pop()  # remove and close existing handlers
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
        if not exists(self._path):
            mkdir(self._path)

        has_log_file = False
        log_level = 20  # default, but overwritten by the level of active loggers
        for i, h in enumerate(self.logger.handlers):
            log_level = h.level
            if hasattr(h, "baseFilename"):
                if dirname(h.baseFilename) != self._path:
                    # remove handler and close file S
                    self.logger.handlers.pop(i).close()
                else:
                    has_log_file = True
                break

        # if not has_log_file:
        new_path = join(self._path, "hydromt.log")
        if overwrite and exists(new_path):
            remove(new_path)

        # if old_path is not None:
        #     old_log_path = join(old_path, "hydromt.log")
        #     rename(old_log_path, new_path)

        add_filehandler(self.logger, new_path, log_level)

    def __repr__(self):
        return f"ModelRoot(path={self._path}, mode={self._mode})"

    def _check_root_exists(self):
        # check directory
        if not isdir(self._path):
            raise IOError(f'model root not found at "{self._path}"')
