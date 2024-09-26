"""Abstract DataSource class."""

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from logging import Logger, getLogger
from os.path import abspath, join
from pathlib import Path, PurePath
from typing import Any, ClassVar, Dict, List, Optional, TypeVar, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from hydromt._typing import DataType, NoDataException, NoDataStrategy, SourceMetadata
from hydromt._utils.uris import _is_valid_url
from hydromt.data_catalog.adapters.data_adapter_base import DataAdapterBase
from hydromt.data_catalog.drivers import BaseDriver
from hydromt.data_catalog.uri_resolvers import ConventionResolver, URIResolver

logger: Logger = getLogger(__name__)

T = TypeVar("T")


class DataSource(BaseModel, ABC):
    """
    A DataSource is a parsed section of a DataCatalog.

    The DataSource, specific for a data type within HydroMT, is responsible for
    validating the input from the DataCatalog, to
    ensure the workflow fails as early as possible. A DataSource has information on
    the driver that the data should be read with, and is responsible for initializing
    this driver.
    """

    model_config = ConfigDict(extra="forbid")

    _used: bool = PrivateAttr(default=False)
    _fallback_driver_read: ClassVar[str]
    _fallback_driver_write: ClassVar[str]

    name: str
    uri: str
    data_adapter: DataAdapterBase
    driver: BaseDriver
    uri_resolver: URIResolver = Field(default_factory=ConventionResolver)
    data_type: ClassVar[DataType]
    root: Optional[str] = Field(
        default=None, exclude=True
    )  # root is already in the catalog.
    version: Optional[Union[str, int, float]] = Field(default=None)
    provider: Optional[str] = Field(default=None)
    metadata: SourceMetadata = Field(default_factory=SourceMetadata)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the DataSource."""
        summ: Dict[str, Any] = self.model_dump(include={"uri"})
        summ.update(
            {
                "data_type": self.__class__.data_type,
                "driver": self.driver.__repr_name__(),
                **self.metadata.model_dump(exclude_unset=True),
            }
        )
        return summ

    def _mark_as_used(self):
        """Mark the data adapter as used."""
        self._used = True

    @model_validator(mode="before")
    @classmethod
    def _validate_data_type(cls, data: Any) -> Any:
        """Pydantic does not check class variables, so it is checked here."""
        if isinstance(data, dict):
            copy_data: dict = deepcopy(data)
            if data_type := copy_data.pop("data_type", None):
                if data_type != cls.data_type:
                    raise ValueError(f"'data_type' must be '{cls.data_type}'.")
        return copy_data

    @model_validator(mode="after")
    def _validate_fs_equal_if_not_set(self) -> DataSource:
        """
        Validate and change the filesystems.

        They have to be equal between driver and uri resolver if they are not set.
        They can be different, but only if set explicitly.
        """
        driver_fs_set = "filesystem" in self.driver.model_fields_set
        uri_res_fs_set = "filesystem" in self.uri_resolver.model_fields_set
        if driver_fs_set ^ uri_res_fs_set:
            if driver_fs_set:
                self.uri_resolver.filesystem = self.driver.filesystem
            else:
                self.driver.filesystem = self.uri_resolver.filesystem
        return self

    @property
    def full_uri(self) -> str:
        """Join root with uri."""
        uri_is_url: bool = _is_valid_url(self.uri)
        if uri_is_url:
            # uri is fully self-describing
            return self.uri
        if not uri_is_url and self.root:
            if _is_valid_url(self.root):
                # use '/' to connect url parts
                return f"{self.root.rstrip('/')}/{self.uri}"
        # Local file, make absolute
        return _abs_path(self.root, self.uri)

    @model_serializer(mode="wrap")
    def _serialize(self, nxt: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        """Serialize data_type."""
        res: Dict[str, Any] = nxt(self)
        res["data_type"] = self.data_type
        return res

    def _get_uri_basename(self, handle_nodata: NoDataStrategy, **query_kwargs) -> str:
        if "{" in self.uri:
            # first resolve any placeholders
            # FIXME: place me in the to_file interface
            uris: List[str] = self.uri_resolver.resolve(
                uri=self.full_uri,
                handle_nodata=handle_nodata,
                **query_kwargs,
            )

            # if multiple_uris, use the first one:
            if len(uris) > 0:
                uri: str = uris[0]
            else:
                raise NoDataException("!")
        else:
            uri: str = self.uri

        basename: Optional[str] = PurePath(uri).name
        if basename is None:
            raise ValueError(f"Failed to get basename of uri: {self.uri}")
        else:
            return basename


def _abs_path(root: Union[Path, str], rel_path: Union[Path, str]) -> str:
    path = Path(str(rel_path))
    if not path.is_absolute():
        if root is not None:
            rel_path = join(root, rel_path)
        path = Path(abspath(rel_path))
    return str(path)
