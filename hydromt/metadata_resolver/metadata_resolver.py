"""Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from hydromt._typing import Bbox, Geom, NoDataStrategy, Predicate, TimeRange

logger: Logger = getLogger(__name__)


class MetaDataResolver(BaseModel, ABC):
    """Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""

    unit_add: Dict[str, Any] = Field(default_factory=dict)
    unit_mult: Dict[str, Any] = Field(default_factory=dict)
    rename: Dict[str, str] = Field(default_factory=dict)

    @abstractmethod
    def resolve(
        self,
        uri: str,
        *,
        timerange: Optional[TimeRange] = None,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0.0,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        zoom_level: int = 0,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Optional[Logger] = logger,
        **kwargs,
    ) -> List[str]:
        """Resolve metadata of data behind a single URI."""
        ...
