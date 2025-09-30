from typing import Any

from fsspec import AbstractFileSystem, filesystem
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    model_serializer,
    model_validator,
)


class FSSpecFileSystem(BaseModel):
    """Pydantic compatible fsspec AbstractFileSystem."""

    protocol: str = "file"
    storage_options: dict[str, Any] = Field(default_factory=dict)

    _fs: AbstractFileSystem = PrivateAttr()

    def __init__(
        self,
        protocol: str = "file",
        storage_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        storage_options = storage_options or {}
        storage_options.update(kwargs)  # allow passing storage options as kwargs
        super().__init__(protocol=protocol, storage_options=storage_options)
        self._fs = filesystem(protocol=self.protocol, **self.storage_options)

    def get_fs(self) -> AbstractFileSystem:
        """Get the underlying fsspec filesystem."""
        return self._fs

    @model_serializer()
    def serialize(self) -> dict[str, Any]:
        """Serialize the filesystem to a dict."""
        fs_dict: dict[str, str] = self.get_fs().to_dict(include_password=False)
        fs_dict.pop("cls", None)  # cls is not required
        if "args" in fs_dict and fs_dict["args"] == []:
            fs_dict.pop("args")  # args is optional
        return fs_dict

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, value: Any) -> Any:
        """Validate input before model initialization."""
        if value is None:
            return FSSpecFileSystem()
        return FSSpecFileSystem.create(value)

    @staticmethod
    def create(input: Any) -> "FSSpecFileSystem":
        """Create an fsspec filesystem from various inputs."""
        if isinstance(input, str):
            # input is protocol
            return FSSpecFileSystem(protocol=input)
        elif isinstance(input, dict):
            if not input:
                return FSSpecFileSystem()
            # input is dict with build args for fsspec filesystem.
            if "protocol" not in input:
                raise ValueError(f"Filesystem dict {input} requires 'protocol'.")
            protocol = input.pop("protocol")
            return FSSpecFileSystem(protocol=protocol, **input)
        elif isinstance(input, AbstractFileSystem):
            protocol = (
                input.protocol[0]
                if isinstance(input.protocol, tuple)
                else input.protocol
            )
            return FSSpecFileSystem(protocol=protocol, **input.storage_options)
        else:
            raise ValueError(f"Unknown filesystem: {input}")
