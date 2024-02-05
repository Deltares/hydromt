"""the new model root class."""

from glob import glob
from logging import FileHandler, Logger, getLogger
from os import PathLike, getcwd, getlogin, makedirs
from os.path import abspath, dirname, isdir, join
from pathlib import Path, PurePosixPath, PureWindowsPath
from platform import system
from typing import List, Optional, Tuple, cast

from hydromt.log import add_filehandler
from hydromt.typing import ModeLike, ModelMode

logger = getLogger(__name__)


class ModelRoot:
    """A class to handle model roots in a cross platform manner."""

    def __init__(
        self,
        path: PathLike,
        mode: ModeLike = "w",
        folders: Optional[List[str]] = None,
        create_dirs: bool = False,
        logger: Logger = logger,
    ):
        self.logger: Logger = logger
        self.folders = folders or ["."]
        self.set_root(path, mode, create_dirs)

    def set_root(self, path: PathLike, mode: ModeLike = "w", create_dirs: bool = False):
        """Initialize the model root.

        In read/append mode a check is done if the root exists.
        In write mode the required model folder structure is created.

        Parameters
        ----------
        root : str, optional
            path to model root
        mode : {"r", "r+", "w"}, optional
            read/append/write mode for model files
        """
        self.set_mode(mode)
        # set path is to take care of cross platform paths
        self._set_path(path)
        if self.is_writing_mode():
            if create_dirs:
                self._setup_folder_structure()
                self._setup_log_file_handlers()
        if self.is_reading_mode():
            self._check_root_exists()

    def _check_root_exists(self):
        # check directory
        if not isdir(self._path):
            raise IOError(f'model root not found at "{self._path}"')

    def _setup_folder_structure(self):
        ignore_ext = [".log", ".yml"]
        for name in self.folders:
            dir_path = join(self._path, name)
            if not isdir(dir_path):
                makedirs(dir_path)
                continue
            # path already exists, check files
            files = glob(join(dir_path, "*.*"))
            files = [Path(file).suffix for file in files]
            files = [ext for ext in files if ext not in ignore_ext]
            if len(files) != 0:
                if self.mode.is_override():
                    self.logger.warning(
                        "Model dir already exists and "
                        f"files might be overwritten: {self._path}."
                    )
                else:
                    msg = (
                        "Model dir already exists and cannot be "
                        + f"overwritten: {self._path}. Use 'mode=w+' to force "
                        + "overwrite existing files."
                    )
                    self.logger.error(msg)
                    raise IOError(msg)

    def _setup_log_file_hanglers(self) -> None:
        # remove old logging file handler and add new filehandler
        # in root if it does not exist
        has_log_file = False
        log_level = 20  # default, but overwritten by the level of active loggers
        for i, handeler in enumerate(self.logger.handlers):
            # make the type checkers a little happier
            handeler = cast(FileHandler, handeler)
            log_level = handeler.level
            if hasattr(handeler, "baseFilename"):
                if dirname(handeler.baseFilename) != self._path:
                    # remove handler and close file
                    self.logger.handlers.pop(i).close()
                else:
                    has_log_file = True
                break
        if not has_log_file:
            new_path = join(self._path, "hydromt.log")
            add_filehandler(self.logger, new_path, log_level)

    def _set_path(self, path: PathLike) -> None:
        """Set the path of the root, adjusting for cross platform where necessary."""
        if self._describes_windows_path(path):
            pwp = PureWindowsPath(path)
            if system() in ["Linux", "Darwin"]:
                if not pwp.is_absolute():
                    # posix path on a windows system
                    abs_path = abspath(join(getcwd(), pwp.as_posix()))
                else:
                    drive = pwp.drive
                    relative_path_part = PureWindowsPath(str(pwp).removeprefix(drive))
                    drive = drive.replace(":", "").lower()
                    abs_path = abspath(
                        join("/mnt", drive, *(relative_path_part.parts[1:]))
                    )
                self._path: Path = Path(abs_path)
            else:
                # windows path on a windows system
                self._path: Path = Path(abspath(pwp))

        elif self._describes_posix_path(path):
            ppp = PurePosixPath(path)
            if system() == "Windows":
                # posix path on a windows system
                parts = ppp.parts
                if self._posix_is_mounted(ppp):
                    # were mounted somewhere so map the second dir to a drive
                    drive = parts[1]
                    relative_path_part = parts[2:]
                    abs_path = abspath(join(f"{drive.upper()}:", *relative_path_part))
                else:
                    abs_ppp = Path(abspath(ppp))
                    # abspath will almost certainly look like /home/user/whatever
                    relative_path_part = abs_ppp.parts[3:]
                    # in windows user dirs are usually titleized
                    abs_path = abspath(
                        join(f"C:\\Users\\{getlogin().title()}", *relative_path_part)
                    )

                self._path: Path = Path(abs_path)

            else:
                # posix path on a posix system
                self._path: Path = Path(abspath(ppp))

    def set_mode(self, mode: ModeLike) -> None:
        """Set the mode of the model."""
        self.mode: ModelMode = ModelMode.from_str_or_mode(mode)

    def is_writing_mode(self) -> bool:
        """Test whether we are in writing mode or not."""
        return self.mode.is_writing_mode()

    def is_reading_mode(self) -> bool:
        """Test whether we are in reading mode or not."""
        return self.mode.is_reading_mode()

    def _posix_is_mounted(self, path) -> bool:
        # check if the first dir is one of the canonical
        # places to mount in linux
        return Path(path).parts[1] in ["mnt", "media"]

    def _count_slashes(self, path) -> Tuple[int, int]:
        """Return the number of each shales in the path as (forward,backward)."""
        str_path = str(path)
        forward_count = str_path.count("/")
        backward_count = str_path.count("\\")

        return forward_count, backward_count

    def _describes_windows_path(self, path: PathLike, logger: Logger = logger) -> bool:
        forward_count, backward_count = self._count_slashes(path)
        # not great, but it's the best we have
        return forward_count < backward_count

    def _describes_posix_path(self, path: PathLike, logger: Logger = logger) -> bool:
        forward_count, backward_count = self._count_slashes(path)
        # not great, but it's the best we have
        return forward_count > backward_count

    def __repr__(self):
        return f"ModelRoot(path={self._path}, mode={self.mode})"
