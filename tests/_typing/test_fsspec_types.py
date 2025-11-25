from typing import Any

import pytest
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from hydromt._compat import HAS_S3FS
from hydromt.typing.fsspec_types import (
    FSSpecFileSystem,
)

TEST_KEYS = ("protocol", "expected_dump", "expected_fs")
TEST_VALUES = [
    ("memory", {"protocol": "memory"}, MemoryFileSystem),
    ("file", {"protocol": "file"}, LocalFileSystem),
]
if HAS_S3FS:
    from s3fs import S3FileSystem

    TEST_VALUES.append(("s3", {"protocol": "s3"}, S3FileSystem))


@pytest.mark.parametrize(TEST_KEYS, TEST_VALUES)
def test_serialize_fs(protocol, expected_dump, expected_fs):
    fs = FSSpecFileSystem(protocol=protocol)
    dump = fs.serialize()
    assert isinstance(fs.get_fs(), expected_fs)
    assert dump == expected_dump, (
        f"Failed to serialize filesystem, expected {expected_dump}, got {dump}"
    )


@pytest.mark.parametrize(TEST_KEYS, TEST_VALUES)
def test_deserializes_fs(protocol, expected_dump, expected_fs):
    fs = FSSpecFileSystem.model_validate(expected_dump)

    assert isinstance(fs.get_fs(), expected_fs)
    assert fs.serialize() == expected_dump, (
        f"Failed to serialize filesystem, expected {expected_dump}, got {fs.serialize()}"
    )
    assert protocol == fs.protocol


@pytest.mark.parametrize(
    "storage_options",
    [
        {},
        {"key": "value"},
        {"auth": {"token": "SomeToken", "other": 123}},
        {"param": 42, "flag": True},
    ],
)
def test_filesystem_storage_options(storage_options: dict[str, Any]):
    fs = FSSpecFileSystem(protocol="file", storage_options=storage_options)
    assert fs.storage_options == storage_options
    dump = fs.serialize()
    for k, v in storage_options.items():
        assert k in dump
        assert dump[k] == v


@pytest.mark.parametrize(
    ("create_input", "expected_fs", "expected_dump", "error_msg"),
    [
        (
            {"protocol": "memory", "max_paths": 50},
            MemoryFileSystem,
            {"protocol": "memory", "max_paths": 50},
            None,
        ),
        ({"max_paths": 50}, None, None, "requires 'protocol'"),
        (LocalFileSystem(), LocalFileSystem, {"protocol": "file"}, None),
        (42, None, None, "Unknown filesystem"),
    ],
)
def test_create_fs(
    create_input: Any,
    expected_fs: type[AbstractFileSystem] | None,
    expected_dump: dict[str, Any] | None,
    error_msg: str | None,
):
    if error_msg:
        with pytest.raises(ValueError, match=error_msg):
            FSSpecFileSystem.create(create_input)
    else:
        fs = FSSpecFileSystem.create(create_input)
        assert expected_fs is not None
        assert isinstance(fs.get_fs(), expected_fs)
        assert fs.serialize() == expected_dump, (
            f"Failed to serialize filesystem, expected {expected_dump}, got {fs.serialize()}"
        )
