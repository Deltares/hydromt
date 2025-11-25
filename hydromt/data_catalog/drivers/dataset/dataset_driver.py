"""Abstract driver to read datasets."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import xarray as xr

from hydromt.data_catalog.drivers.base_driver import (
    BaseDriver,
)
from hydromt.error import NoDataStrategy

logger = logging.getLogger(__name__)


class DatasetDriver(BaseDriver, ABC):
    """Abstract Driver to read Datasets."""

    @abstractmethod
    def read(
        self, uris: list[str], *, handle_nodata: NoDataStrategy = NoDataStrategy.RAISE
    ) -> xr.Dataset:
        """
        Read data from one or more URIs into an xarray Dataset.

        This abstract method defines the interface for all dataset drivers. Subclasses
        should implement data loading logic appropriate for the format being read.

        Parameters
        ----------
        uris : list[str]
            List of URIs to read data from.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle missing or empty data. Default is NoDataStrategy.RAISE.

        Returns
        -------
        xr.Dataset
            The loaded dataset.
        """
        ...

    @abstractmethod
    def write(
        self,
        path: Path | str,
        data: xr.Dataset,
        *,
        write_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write an xarray Dataset to disk.

        This abstract method defines the interface for all Dataset-based drivers.
        Subclasses should implement logic for writing datasets in specific formats
        (e.g., NetCDF, Zarr).

        Parameters
        ----------
        path : Path | str
            Destination path or URI where the Dataset should be written.
        data : xr.Dataset
            The Dataset to write.
        write_kwargs : dict[str, Any], optional
            Additional keyword arguments to pass to the underlying xarray write function
            (e.g., `to_zarr`, `to_netcdf`). Default is None.

        Returns
        -------
        Path
            The path where the data was written.

        """
        ...
