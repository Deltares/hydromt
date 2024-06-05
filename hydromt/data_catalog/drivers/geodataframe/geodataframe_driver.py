"""Driver for GeoDataFrames."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, Optional

import geopandas as gpd

from hydromt._typing import Geom, SourceMetadata, StrPath
from hydromt._typing.error import NoDataStrategy
from hydromt.data_catalog.drivers import BaseDriver

logger: Logger = getLogger(__name__)


class GeoDataFrameDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDataFrames."""

    def read(
        self,
        uri: str,
        *,
        mask: Optional[Geom] = None,
        variables: Optional[List[str]] = None,
        predicate: str = "intersects",
        metadata: Optional[SourceMetadata] = None,
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
    ) -> gpd.GeoDataFrame:
        """
        Read in any compatible data source to a geopandas `GeoDataFrame`.

        args:
            mask: Optional[Geom]. Mask for features to match the predicate, preferably
                in the same CRS.
        """
        # Merge static kwargs from the catalog with dynamic kwargs from the query.
        uris = self.metadata_resolver.resolve(
            uri,
            self.filesystem,
            mask=mask,
            variables=variables,
            handle_nodata=handle_nodata,
        )
        gdf = self.read_data(
            uris,
            mask=mask,
            predicate=predicate,
            variables=variables,
            metadata=metadata,
            logger=logger,
            handle_nodata=handle_nodata,
        )
        return gdf

    @abstractmethod
    def read_data(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        predicate: str = "intersects",
        variables: Optional[List[str]] = None,
        metadata: Optional[SourceMetadata] = None,
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> gpd.GeoDataFrame:
        """Read in any compatible data source to a geopandas `GeoDataFrame`."""
        ...

    def write(
        self,
        path: StrPath,
        gdf: gpd.GeoDataFrame,
        **kwargs,
    ) -> None:
        """
        Write out a GeoDataFrame to file.

        Not all drivers should have a write function, so this method is not
        abstract.

        args:
        """
        raise NotImplementedError(
            f"Writing using driver '{self.name}' is not supported."
        )
