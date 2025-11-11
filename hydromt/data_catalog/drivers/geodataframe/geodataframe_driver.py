"""Driver for GeoDataFrames."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import geopandas as gpd

from hydromt.data_catalog.drivers import BaseDriver
from hydromt.error import NoDataStrategy
from hydromt.typing import SourceMetadata

logger = logging.getLogger(__name__)


class GeoDataFrameDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        metadata: SourceMetadata | None = None,
        mask: Any = None,
        variables: str | list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Read geospatial data into a GeoDataFrame.

        Parameters
        ----------
        uris : list[str]
            List of URIs to read data from.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle missing or empty data. Default is NoDataStrategy.RAISE.
        metadata : SourceMetadata | None, optional
            Optional metadata object describing the dataset source (e.g. CRS).
        mask : Any, optional
            Optional spatial mask to filter the data. The mask can be a geometry, GeoDataFrame,
            or any geometry-like object depending on driver support.
        variables : str | list[str] | None, optional
            Optional variable(s) or column(s) to read from the source file.

        Returns
        -------
        gpd.GeoDataFrame
            The loaded geospatial data.
        """
        ...

    @abstractmethod
    def write(
        self,
        path: Path | str,
        data: gpd.GeoDataFrame,
        *,
        write_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write a GeoDataFrame to disk.

        This abstract method defines the interface for all geospatial data drivers.
        Subclasses should implement logic for writing to supported vector formats
        (e.g., GeoPackage, Shapefile, GeoJSON).

        Parameters
        ----------
        path : Path | str
            Destination path or URI where the GeoDataFrame will be written.
        data : gpd.GeoDataFrame
            The GeoDataFrame to write.
        write_kwargs : dict[str, Any], optional
            Additional keyword arguments passed to the underlying write function
            (e.g., `pyogrio.write_dataframe`). Default is None.

        Returns
        -------
        Path
            The path where the data was written.

        """
        ...
