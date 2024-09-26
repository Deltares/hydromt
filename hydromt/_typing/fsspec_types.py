from typing import Any, Dict, Iterable

from fsspec import AbstractFileSystem, filesystem
from pydantic import PlainSerializer, PlainValidator
from typing_extensions import Annotated


def validate_filesystem(
    _fs: Any,
) -> AbstractFileSystem:
    if isinstance(_fs, str):
        # _fs is protocol
        return filesystem(_fs)
    elif isinstance(_fs, dict):
        # _fs is dict with build args for fsspec filesystem.
        if protocol := _fs.pop("protocol", None):
            # fsspec to_dict gets some args
            args: Iterable[Any] = _fs.pop("args", [])
            return filesystem(protocol, *args, **_fs)
        else:
            raise ValueError(f"Filesystem dict {_fs} requires 'protocol'.")
    elif isinstance(_fs, AbstractFileSystem):
        # _fs is already deserialized
        return _fs
    else:
        raise ValueError(f"Unknown filesystem: {_fs}")


def serialize_filesystem(_fs: AbstractFileSystem) -> Dict[str, Any]:
    fs_dict: Dict[str, str] = _fs.to_dict(
        include_password=False
    )  # do not include any passwords.
    # cls is not required
    fs_dict.pop("cls")
    if not fs_dict.get("args"):
        fs_dict.pop("args")
    return fs_dict


FS = Annotated[
    AbstractFileSystem,
    PlainValidator(validate_filesystem),
    PlainSerializer(serialize_filesystem),
]
