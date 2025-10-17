"""Driver for handling IO of GeoDatasets."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
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
from hydromt.typing import Geom, Predicate, SourceMetadata

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
        open_kwargs: dict[str, Any] | None = None,
        mask: Geom | None = None,
        predicate: Predicate = "intersects",
        metadata: SourceMetadata | None = None,
    ) -> xr.Dataset | None:
        """
        Read in data to an xarray Dataset.

        Parameters
        ----------
        uris : list[str]
            List of URIs to read data from.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle missing data. Default is NoDataStrategy.RAISE.
        open_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments to pass to the underlying open function. Default is None.
        mask : Geom | None, optional
            Optional spatial mask to clip the dataset.
        predicate : Predicate, optional
            Spatial predicate for filtering geometries. Default is "intersects".
        metadata : SourceMetadata | None, optional
            Optional metadata object to attach to the loaded dataset.

        Returns
        -------
        xr.Dataset | None
            The dataset read from the source, or None if no data found and strategy allows.
        """
        ...

    @abstractmethod
    def write(
        self,
        path: Path | str,
        data: xr.Dataset,
        *,
        write_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """
        Write a GeoDataset to disk.

        Parameters
        ----------
        path : Path | str
            Destination path or URI where the dataset will be written. The path should
            have a supported extension depending on the concrete driver implementation.
        data : xr.Dataset
            The xarray Dataset to write.
        write_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments to pass to the underlying write function.
            These may include encoding options for NetCDF, or mode/format options for Zarr.
            Default is None.

        Returns
        -------
        str
            The path to the written dataset.

        """
        ...
