"""Pydantic compatible fsspec AbstractFileSystem type."""

from typing import Any

import fsspec
from fsspec import AbstractFileSystem, filesystem
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    model_serializer,
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

    def get_fsmap(
        self, root: str | None = None, storage_options: dict[str, Any] | None = None
    ) -> fsspec.mapping.FSMap:
        """Get the underlying fsspec FSMap.

        Parameters
        ----------
        root : str | None, optional
            Root path for the FSMap.
        storage_options : dict[str, Any] | None, optional
            Additional storage options to merge with this filesystem's own
            storage options when building the mapper. Useful for driver-level
            options (e.g. set under a driver's ``options`` rather than its
            ``filesystem``) that weren't available at filesystem construction
            time. Default is None, which reuses the filesystem built at
            construction time.
        """
        if storage_options:
            fs = filesystem(
                protocol=self.protocol, **{**self.storage_options, **storage_options}
            )
            return fs.get_mapper(root=root)
        return self._fs.get_mapper(root=root)

    @model_serializer()
    def serialize(self, include_credentials: bool = False) -> dict[str, Any]:
        """Serialize the filesystem to a dict.

        Parameters
        ----------
        include_credentials:
            Whether to include passwords/secrets in the serialized dict.
        """
        fs_dict: dict[str, str] = self.get_fs().to_dict(
            include_password=include_credentials
        )
        fs_dict.pop("cls", None)  # cls is not required
        if "args" in fs_dict and fs_dict["args"] == []:
            fs_dict.pop("args")  # args is optional
        return fs_dict

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
            return FSSpecFileSystem(protocol=protocol, storage_options=input)
        elif isinstance(input, AbstractFileSystem):
            protocol = (
                input.protocol[0]
                if isinstance(input.protocol, tuple)
                else input.protocol
            )
            return FSSpecFileSystem(
                protocol=protocol, storage_options=input.storage_options
            )
        else:
            raise ValueError(f"Unknown filesystem: {input}")
