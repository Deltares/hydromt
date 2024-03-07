"""Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""
from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, Optional

import geopandas as gpd
from pydantic import BaseModel, Field

from hydromt._typing import Bbox, NoDataStrategy, Predicate, TimeRange
from hydromt.data_adapter.harmonization_settings import HarmonizationSettings

logger: Logger = getLogger(__name__)


class MetaDataResolver(BaseModel, ABC):
    """Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""

    harmonization_settings: HarmonizationSettings = Field(
        default_factory=HarmonizationSettings
    )

    @abstractmethod
    def resolve(
        self,
        uri: str,
        *,
        timerange: Optional[TimeRange] = None,
        bbox: Optional[Bbox] = None,
        mask: Optional[gpd.GeoDataFrame] = None,
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
