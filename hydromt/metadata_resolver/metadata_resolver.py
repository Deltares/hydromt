"""Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, Optional

from fsspec import AbstractFileSystem
from pydantic import BaseModel, ConfigDict

from hydromt._typing import Geom, NoDataStrategy, TimeRange, ZoomLevel

logger: Logger = getLogger(__name__)


class MetaDataResolver(BaseModel, ABC):
    """Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""

    model_config = ConfigDict(extra="forbid")

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
    ) -> List[str]:
        """Resolve metadata of data behind a single URI."""
        ...
