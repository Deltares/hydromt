"""Driver for handling IO of GeoDatasets."""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import List, Optional

import xarray as xr

from hydromt._typing import Geom, SourceMetadata, StrPath, TimeRange
from hydromt._typing.error import NoDataStrategy
from hydromt._typing.type_def import Predicate
from hydromt.data_catalog.drivers import BaseDriver

logger = getLogger(__name__)


class GeoDatasetDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDatasets."""

    @abstractmethod
    def read(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
    ) -> Optional[xr.Dataset]:
        """
        Read in any compatible data source to an xarray Dataset.

        args:
        """
        ...

    def write(
        self,
        path: StrPath,
        ds: xr.Dataset,
        **kwargs,
    ) -> str:
        """
        Write out a GeoDataset to file.

        Not all drivers should have a write function, so this method is not
        abstract.

        args:
        """
        raise NotImplementedError(
            f"Writing using driver '{self.name}' is not supported."
        )
