"""Driver for GeoDataFrames."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import geopandas as gpd

from hydromt.data_catalog.drivers import BaseDriver
from hydromt.error import NoDataStrategy
from hydromt.typing import SourceMetadata, StrPath

logger = logging.getLogger(__name__)


class GeoDataFrameDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        kwargs_for_open: dict[str, Any] | None = None,
        metadata: SourceMetadata | None = None,
        mask: Any = None,
        predicate: str = "intersects",
        variables: str | list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """Read in any compatible data source to a geopandas `GeoDataFrame`."""
        ...

    def write(
        self,
        path: StrPath,
        gdf: gpd.GeoDataFrame,
        **kwargs,
    ) -> str:
        """
        Write out a GeoDataFrame to file.

        Not all drivers should have a write function, so this method is not
        abstract.

        args:
        """
        raise NotImplementedError(
            f"Writing using driver '{self.name}' is not supported."
        )
