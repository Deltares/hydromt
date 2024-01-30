"""the new model root class."""

from logging import Logger, getLogger
from os import PathLike, getcwd, getlogin
from os.path import abspath, join
from pathlib import Path, PurePosixPath, PureWindowsPath
from platform import system
from typing import Tuple

from hydromt.typing import ModeLike, ModelMode

logger = getLogger(__name__)


class ModelRoot:
    """A class to handle model roots in a cross platform manner."""

    def __init__(self, path: PathLike, mode: ModeLike = "w"):
        self.set_path(path)
        self.set_mode(mode)

    def set_path(self, path: PathLike) -> None:
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

    def assert_writing_mode(self) -> bool:
        """Test whether we are in writing mode or not."""
        return self.mode.is_writing_mode()

    def assert_reading_mode(self) -> bool:
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


# def set_root(self, root: Optional[str], mode: Optional[str] = "w"):
#         """Initialize the model root.

#         In read/append mode a check is done if the root exists.
#         In write mode the required model folder structure is created.

#         Parameters
#         ----------
#         root : str, optional
#             path to model root
#         mode : {"r", "r+", "w"}, optional
#             read/append/write mode for model files
#         """
#         ignore_ext = set([".log", ".yml"])
#         if mode not in ["r", "r+", "w", "w+"]:
#             raise ValueError(
#                 f'mode "{mode}" unknown, select from "r", "r+", "w" or "w+"'
#             )
#         self._root = root if root is None else abspath(root)
#         if self._root is not None:
#             if self._write:
#                 for name in self._FOLDERS:
#                     path = join(self._root, name)
#                     if not isdir(path):
#                         os.makedirs(path)
#                         continue
#                     # path already exists check files
#                     fns = glob.glob(join(path, "*.*"))
#                     exts = set([os.path.splitext(fn)[1] for fn in fns])
#                     exts -= ignore_ext
#                     if len(exts) != 0:
#                         if mode.endswith("+"):
#                             self.logger.warning(
#                                 "Model dir already exists and "
#                                 f"files might be overwritten: {path}."
#                             )
#                         else:
#                             msg = (
#                                 "Model dir already exists and cannot be "
#                                 + f"overwritten: {path}. Use 'mode=w+' to force "
#                                 + "overwrite existing files."
#                             )
#                             self.logger.error(msg)
#                             raise IOError(msg)
#             # check directory
#             elif not isdir(self._root):
#                 raise IOError(f'model root not found at "{self._root}"')
#             # remove old logging file handler and add new filehandler
#             # in root if it does not exist
#             has_log_file = False
#             log_level = 20  # default, but overwritten by the level of active loggers
#             for i, h in enumerate(self.logger.handlers):
#                 log_level = h.level
#                 if hasattr(h, "baseFilename"):
#                     if dirname(h.baseFilename) != self._root:
#                         # remove handler and close file
#                         self.logger.handlers.pop(i).close()
#                     else:
#                         has_log_file = True
#                     break
#             if not has_log_file:
#                 new_path = join(self._root, "hydromt.log")
#                 log.add_filehandler(self.logger, new_path, log_level)
