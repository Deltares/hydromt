"""A module to handle paths from different platforms in a cross platform compatible manner."""
from os import PathLike, getcwd, getlogin
from os.path import abspath, exists, join
from pathlib import Path, PurePosixPath, PureWindowsPath
from platform import system
from typing import List, Optional, Tuple


def _posix_is_mounted(path) -> bool:
    # check if the first dir is one of the canonical
    # places to mount in linux
    return Path(path).parts[1] in ["mnt", "media"]


def _count_slashes(path) -> Tuple[int, int]:
    """Return the number of each shales in the path as (forward,backward)."""
    str_path = str(path)
    forward_count = str_path.count("/")
    backward_count = str_path.count("\\")

    return forward_count, backward_count


def _describes_windows_path(path: PathLike) -> bool:
    forward_count, backward_count = _count_slashes(path)
    # not great, but it's the best we have
    return forward_count < backward_count


def _describes_posix_path(path: PathLike) -> bool:
    forward_count, backward_count = _count_slashes(path)
    # not great, but it's the best we have
    return forward_count > backward_count


def make_path_abs_and_cross_platform(path: PathLike) -> Path:
    r"""
    Return an absolute path accodring to the conventions of the platform we're running on.

    Dealing with paths that describe a file path for a platrom we're not on (e.g. handeling Windows paths on a POSIX system)
    easily leads to trouble. This function will apply the following assumptions:

    - Paths that describe paths on the same platform are handled like normally
    - on POSIX platforms a windows path like C:\\to\\somehwere maps to /mnt/c/to/somewhere. for linux this is the cannonical place to mount things like network shares.
    - Relative Windows paths on POSIX platforms are simply converted to their posix/linux equivalent and used like normal
    - on Windows a POSIX path like /mnt/d/to/somewhere will map to D:\\to\\somewhere mostly as an inverse as the previous point.
    - A path like /home/user/to/somewhere will map to C:\\Users\\{login_name_of_user}\\to\\somewhere. so keep in mind it will map to the
        home dir of the current user, not the user specified in the path.

    Arguments:
    ----------
    path: PathLike
        The path to convert

    Returns
    -------
    Path:
        The absolute path according to the convention of the platform we're on pointing to the same place as the provided path

    """
    if _describes_windows_path(path):
        pwp = PureWindowsPath(path)
        if system() in ["Linux", "Darwin"]:
            if not pwp.is_absolute():
                # posix path on a windows system
                abs_path = abspath(join(getcwd(), pwp.as_posix()))
            else:
                drive = pwp.drive
                relative_path_part = PureWindowsPath(str(pwp).removeprefix(drive))
                drive = drive.replace(":", "").lower()
                abs_path = abspath(join("/mnt", drive, *(relative_path_part.parts[1:])))
            return Path(abs_path)
        else:
            # windows path on a windows system
            return Path(abspath(pwp))

    elif _describes_posix_path(path):
        ppp = PurePosixPath(path)
        if system() == "Windows":
            # posix path on a windows system
            parts = ppp.parts
            if _posix_is_mounted(ppp):
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

            return Path(abs_path)
        else:
            # posix path on a posix system
            return Path(abspath(ppp))
    else:
        raise ValueError(f"Could not dermine what kind of paths is described by {path}")


def parse_relpath(cfdict: dict, root: Path) -> dict:
    """Parse string/path value to relative path if possible."""

    def _relpath(value, root):
        if isinstance(value, str) and str(Path(value)).startswith(str(root)):
            value = Path(value)
        if isinstance(value, Path):
            try:
                rel_path = value.relative_to(root)
                value = str(rel_path).replace("\\", "/")
            except ValueError:
                pass  # `value` path is not relative to root
        return value

    # loop through n-level of dict
    for key, val in cfdict.items():
        if isinstance(val, dict):
            cfdict[key] = parse_relpath(val, root)
        else:
            cfdict[key] = _relpath(val, root)
    return cfdict


def parse_abspath(
    cfdict: dict, root: Path, skip_abspath_sections: Optional[List] = None
) -> dict:
    """Parse string value to absolute path from config file."""
    skip_abspath_sections = skip_abspath_sections or ["setup_config"]

    def _abspath(value, root):
        if exists(join(root, value)):
            value = Path(abspath(join(root, value)))
        return value

    # loop through n-level of dict
    for key, val in cfdict.items():
        if isinstance(val, dict):
            if key not in skip_abspath_sections:
                cfdict[key] = parse_abspath(val, root)
        elif isinstance(val, list) and all([isinstance(v, str) for v in val]):
            cfdict[key] = [_abspath(v, root) for v in val]
        elif isinstance(val, str):
            cfdict[key] = _abspath(val, root)
    return cfdict
