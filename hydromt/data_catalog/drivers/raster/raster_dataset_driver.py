"""Driver for RasterDatasets."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import xarray as xr

from hydromt.data_catalog.drivers.base_driver import (
    BaseDriver,
)
from hydromt.error import NoDataStrategy
from hydromt.typing import (
    Geom,
    SourceMetadata,
    Variables,
    Zoom,
)

logger = logging.getLogger(__name__)


class RasterDatasetDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        mask: Geom | None = None,
        variables: Variables | None = None,
        zoom: Zoom | None = None,
        chunks: dict[str, Any] | None = None,
        metadata: SourceMetadata | None = None,
    ) -> xr.Dataset:
        """
        Read raster data from one or more URIs into an xarray Dataset.

        This abstract method defines the common interface for raster-based drivers.
        Implementations must return an xarray Dataset representing the loaded raster data.

        Parameters
        ----------
        uris : list[str]
            List of file URIs to read from.
        handle_nodata : NoDataStrategy, optional
            Strategy for handling missing or empty data. Default is NoDataStrategy.RAISE.
        mask : Geom | None, optional
            Optional geometry to spatially mask or clip the data. Default is None.
        variables : Variables | None, optional
            List of variable names or bands to read from the dataset. Default is None.
        zoom : Zoom | None, optional
            Optional zoom level or resolution control for multi-resolution raster data. Default is None.
        chunks : dict[str, Any] | None, optional
            Chunking configuration for Dask-based reading. Default is None.
        metadata : SourceMetadata | None, optional
            Metadata about the source such as CRS, nodata values, and zoom levels. Default is None.

        Returns
        -------
        xr.Dataset
            The loaded raster dataset.

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
        Write a RasterDataset to a local file.

        Parameters
        ----------
        path : Path | str
            Destination path or URI where the raster dataset will be written. The path
            should have a supported extension depending on the concrete driver implementation.
        data : xr.Dataset
            The xarray Dataset representing the raster data to write.
        write_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments to pass to the underlying write function.
            Default is None.

        Returns
        -------
        Path
            The path to the written raster dataset.
        """
        ...
