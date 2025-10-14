"""Driver for handling IO of GeoDatasets."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import xarray as xr
from pydantic import Field

from hydromt.data_catalog.drivers.base_driver import (
    DRIVER_OPTIONS_DESCRIPTION,
    BaseDriver,
    DriverOptions,
)
from hydromt.data_catalog.drivers.preprocessing import get_preprocessor
from hydromt.error import NoDataStrategy
from hydromt.typing import Geom, Predicate, SourceMetadata, StrPath, TimeRange

logger = logging.getLogger(__name__)


class GeoDatasetOptions(DriverOptions):
    """Options for GeoDatasetVectorDriver."""

    preprocess: str | None = None
    """Name of preprocessor to apply on geodataset after reading. Available preprocessors include: round_latlon, to_datetimeindex, remove_duplicates, harmonise_dims. See their docstrings for details."""

    def get_preprocessor(self) -> Callable | None:
        """Get the preprocessor function."""
        if self.preprocess is None:
            return None
        return get_preprocessor(self.preprocess)


class GeoDatasetDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDatasets."""

    options: GeoDatasetOptions = Field(
        default_factory=GeoDatasetOptions, description=DRIVER_OPTIONS_DESCRIPTION
    )

    @abstractmethod
    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        kwargs_for_open: dict[str, Any] | None = None,
        mask: Geom | None = None,
        predicate: Predicate = "intersects",
        variables: list[str] | None = None,
        time_range: TimeRange | None = None,
        metadata: SourceMetadata | None = None,
    ) -> xr.Dataset | None:
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
