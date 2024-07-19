import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from pydantic import BaseModel

from hydromt._typing.fsspec_types import FS, serialize_filesystem, validate_filesystem


class TestSerializeFileSystem:
    def test_serializes(self):
        fs = MemoryFileSystem()
        assert serialize_filesystem(fs) == {"protocol": "memory"}

    def test_serializes_fs_with_multiple_names(self):
        fs = LocalFileSystem()
        assert serialize_filesystem(fs) == {"protocol": "file"}


class TestValidateFileSystem:
    def test_validates_str(self):
        assert isinstance(validate_filesystem("local"), LocalFileSystem)

    def test_validates_class(self):
        assert isinstance(
            validate_filesystem(LocalFileSystem()),
            LocalFileSystem,
        )

    def test_validates_unknown(self):
        with pytest.raises(ValueError, match="Unknown filesystem"):
            validate_filesystem(42)

    def test_validates_dict(self):
        fs = validate_filesystem({"protocol": "memory", "max_paths": 50})
        assert isinstance(fs, MemoryFileSystem)

    def test_validates_dict_no_protocol(self):
        with pytest.raises(ValueError, match="requires 'protocol'"):
            validate_filesystem({"max_paths": 50})


class MyModel(BaseModel):
    fs: FS


def test_deserializes_model():
    MyModel.model_validate({"fs": {"protocol": "s3", "anon": True}})
