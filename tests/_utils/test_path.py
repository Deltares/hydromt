from pathlib import Path

import pytest

from hydromt._utils.path import _check_directory


def test__check_directory(tmp_path: Path):
    # Call the function
    p = _check_directory(path=tmp_path)

    # Assert the output
    assert p == tmp_path


def test__check_directory_root(tmp_path: Path):
    # Call the function
    p = _check_directory(path="foo", root=tmp_path, fail=False)

    # Assert the output
    assert p == Path(tmp_path, "foo")
    assert p.exists()


def test__check_directory_errors(tmp_path: Path):
    # Call the function while the directory doesn't exist and fail is True
    p = Path(tmp_path, "foo")
    with pytest.raises(
        IOError,
        match=f"{p.as_posix()} does not exist",
    ):
        _ = _check_directory(path="foo", root=tmp_path, fail=True)
