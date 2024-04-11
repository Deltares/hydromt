from typing import Any, Dict, Tuple, Union

from fsspec import AbstractFileSystem, filesystem
from pydantic import PlainSerializer, PlainValidator
from typing_extensions import Annotated


def validate_filesystem(
    _fs: Union[AbstractFileSystem, Dict[str, Any], str],
) -> AbstractFileSystem:
    if isinstance(_fs, str):
        return filesystem(_fs)
    elif isinstance(_fs, dict):
        if protocol := _fs.pop("protocol", None):
            return filesystem(protocol=protocol, **_fs)
        else:
            raise ValueError(f"Filesystem dict {_fs} requires 'protocol'.")
    elif isinstance(_fs, AbstractFileSystem):
        return _fs
    else:
        raise ValueError(f"Unknown filesystem: {_fs}")


def serialize_filesystem(_fs: AbstractFileSystem) -> str:
    pr: Union[str, Tuple[str, ...]] = _fs.protocol
    if isinstance(pr, str):
        return pr
    else:
        return pr[0]  # return first of the protocol names as name


FS = Annotated[
    AbstractFileSystem,
    PlainValidator(validate_filesystem),
    PlainSerializer(serialize_filesystem),
]
