from pathlib import Path

import pytest

from hydromt.model.mode import ModelMode
from hydromt.model.root import ModelRoot


def test_model_root(tmp_path: Path):
    # Set up the root
    root = ModelRoot(path=Path(tmp_path, "foo"), mode="w")

    # Assert some basic stuff
    p = Path(tmp_path, "foo")
    assert root.path == p
    assert root.path.exists()
    assert root.mode == ModelMode("w")


def test_model_root__assert_read(tmp_path: Path):
    # Set up the root
    root = ModelRoot(path=tmp_path, mode="r")

    # Asserting read should be fine
    root._assert_read_mode()

    # Asserting write mode should error
    with pytest.raises(
        IOError,
        match="Model opened in read-only mode",
    ):
        root._assert_write_mode()


def test_model_root__assert_write(tmp_path: Path):
    # Set up the root
    root = ModelRoot(path=tmp_path, mode="w")

    # Asserting write mode should be fine
    root._assert_write_mode()

    # Asserting read mode should error
    with pytest.raises(
        IOError,
        match="Model opened in write-only mode",
    ):
        root._assert_read_mode()


def test_model_root__assert_append(tmp_path: Path):
    # Set up the root
    root = ModelRoot(path=tmp_path, mode="r+")

    # Asserting both modes should be fine
    root._assert_read_mode()
    root._assert_write_mode()


def test_model_root__clean_up(tmp_path: Path):
    # Set up the root
    p = Path(tmp_path, "foo")
    root = ModelRoot(path=p, mode="w")
    # Assert it's there
    assert p.exists()

    # Call the method
    root._cleanup()
    # Assert the directory is gone
    assert not p.exists()


def test_model_root_read_check(tmp_path: Path):
    # Set up the root
    root = ModelRoot(path=tmp_path, mode="r")

    # Assert the mode
    assert root.is_reading_mode()
    assert not root.is_writing_mode()
    assert not root.is_override_mode()


def test_model_root_write_check(tmp_path: Path):
    # Set up the root
    root = ModelRoot(path=tmp_path, mode="w")

    # Assert the mode
    assert not root.is_reading_mode()
    assert root.is_writing_mode()
    assert not root.is_override_mode()


def test_model_root_append_check(tmp_path: Path):
    # Set up the root
    root = ModelRoot(path=tmp_path, mode="r+")

    # Assert the mode
    assert root.is_reading_mode()
    assert root.is_writing_mode()
    assert root.is_override_mode()


def test_model_root_overwrite_check(tmp_path: Path):
    # Set up the root
    root = ModelRoot(path=tmp_path, mode="w+")

    # Assert the mode
    assert not root.is_reading_mode()
    assert root.is_writing_mode()
    assert root.is_override_mode()


def test_model_root_set(tmp_path: Path):
    # Create a new directory
    p = Path(tmp_path, "foo")
    p.mkdir(parents=True)
    # Set up the root
    root = ModelRoot(path=tmp_path, mode="r")
    # Assert current root
    assert root.path == tmp_path
    assert root.mode == ModelMode("r")

    # Set a new root, also in read mode
    _ = root.set(path=p)
    # Assert new root
    assert root.path == p
    assert root.mode == ModelMode("r")


def test_model_root_set_clean(tmp_path: Path):
    # Create a new directory
    p1 = Path(tmp_path, "foo")
    # Set up the root
    root = ModelRoot(path=p1, mode="w")
    # Assert it's there
    assert root.path == p1
    assert p1.exists()

    # Change while foo is still empty
    p2 = Path(tmp_path, "bar")
    root.set(path=p2, mode="w")
    # Assert p2 exists but p1 is deleted
    assert root.path == p2
    assert p2.exists()
    assert not p1.exists()


def test_model_root_set_mode(tmp_path: Path):
    # Set up the root
    root = ModelRoot(path=tmp_path, mode="r")
    # Assert current root
    assert root.path == tmp_path
    assert root.mode == ModelMode("r")

    # Set a new root, also in read mode
    p = Path(tmp_path, "foo")
    _ = root.set(path=p, mode="w")
    # Assert new root
    assert root.path == p
    assert root.mode == ModelMode("w")


def test_model_root_set_errors(tmp_path: Path):
    # Set up the root
    root = ModelRoot(path=tmp_path, mode="w")
    # Assert current root
    assert root.path == tmp_path
    assert root.mode == ModelMode("w")

    # Set the root to a directory that does not exist in read mode
    p = Path(tmp_path, "foo")
    with pytest.raises(
        IOError,
        match=f"{p.as_posix()} does not exist",
    ):
        _ = root.set(path=p, mode="r")
