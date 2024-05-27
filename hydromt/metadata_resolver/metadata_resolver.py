"""Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import Any, ClassVar, Dict, List, Optional

from fsspec import AbstractFileSystem
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from hydromt._typing import Geom, NoDataStrategy, TimeRange, ZoomLevel

logger: Logger = getLogger(__name__)


class MetaDataResolver(BaseModel, ABC):
    """Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""

    model_config = ConfigDict(extra="forbid")
    name: ClassVar[str]

    @model_serializer(mode="wrap")
    def _serialize(self, nxt: SerializerFunctionWrapHandler) -> Any:
        """Also serialize name."""
        res = nxt(self)
        res["name"] = self.name
        return res

    @abstractmethod
    def resolve(
        self,
        uri: str,
        fs: AbstractFileSystem,
        *,
        time_range: Optional[TimeRange] = None,
        mask: Optional[Geom] = None,
        variables: Optional[List[str]] = None,
        zoom_level: Optional[ZoomLevel] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Optional[Logger] = logger,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Resolve metadata of data behind a single URI."""
        ...
