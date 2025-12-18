"""Driver for reading in GeoDataFrames from tabular formats."""

import logging
from pathlib import Path
from typing import Any, ClassVar

import geopandas as gpd
from pydantic import Field

from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers.base_driver import (
    DRIVER_OPTIONS_DESCRIPTION,
    DriverOptions,
)
from hydromt.data_catalog.drivers.geodataframe.geodataframe_driver import (
    GeoDataFrameDriver,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.readers import open_vector_from_table
from hydromt.typing.metadata import SourceMetadata

logger = logging.getLogger(__name__)

# possible labels for the x and y dimensions
X_DIM_LABELS = ("x", "longitude", "lon", "long")
Y_DIM_LABELS = ("y", "latitude", "lat")


class GeoDataFrameTableOptions(DriverOptions):
    """Options for the GeoDataFrameTableDriver."""

    x_dim: str | None = None
    y_dim: str | None = None


class GeoDataFrameTableDriver(GeoDataFrameDriver):
    """
    Driver for GeoDataFrame from tabular formats: ``geodataframe_table``.

    Supports reading point geometries from csv, excel (xls, xlsx), and parquet files
    using a combination of the pandas and geopandas libraries.

    """

    name: ClassVar[str] = "geodataframe_table"
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".csv", ".xlsx", ".xls", ".parquet"}

    options: GeoDataFrameTableOptions = Field(
        default_factory=GeoDataFrameTableOptions, description=DRIVER_OPTIONS_DESCRIPTION
    )

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
        Read tabular geospatial data (CSV, Excel, or Parquet) into a GeoDataFrame.

        Supports reading point geometries from tabular sources using latitude and longitude
        columns. Coordinates are mapped to geometry based on configured dimension names or
        detected automatically from the column headers.

        Parameters
        ----------
        uris : list[str]
            List of URIs to read data from. Only one file is supported per read operation.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle missing or empty data. Default is NoDataStrategy.RAISE.
        metadata : SourceMetadata | None, optional
            Optional metadata object describing the dataset source (e.g. CRS).
        mask : Any, optional
            Unused in this driver. Present for interface consistency.
        variables : str | list[str] | None, optional
            Unused in this driver. Present for interface consistency.

        Returns
        -------
        gpd.GeoDataFrame
            The loaded geospatial data.

        Raises
        ------
        ValueError
            If multiple URIs are provided.

        Warning
        -------
        The `mask` and `variables` parameters are not used directly in this driver, but are included
        for consistency with the GeoDataFrameDriver interface.
        """
        _warn_on_unused_kwargs(
            self.__class__.__name__, {"mask": mask, "variables": variables}
        )

        if not metadata:
            metadata = SourceMetadata()

        if len(uris) > 1:
            raise ValueError(
                "DataFrame: Reading multiple files with the "
                f"{self.__class__.__name__} driver is not supported."
            )

        _uri: str = uris[0]
        gdf = open_vector_from_table(
            path=_uri,
            x_dim=self.options.x_dim,
            y_dim=self.options.y_dim,
            crs=metadata.crs,
            **self.options.get_kwargs(),
        )
        if gdf.index.size == 0:
            exec_nodata_strat(
                f"No data from {self.name} driver for file uris: {', '.join(uris)}.",
                strategy=handle_nodata,
            )
            return None  # handle_nodata == ignore
        return gdf

    def write(
        self,
        path: Path | str,
        data: gpd.GeoDataFrame,
        *,
        write_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write a GeoDataFrame to disk.

        Writing is not supported for this driver, as tabular sources (CSV, Excel,
        Parquet) are typically read-only in this context. This method is included
        to maintain consistency with the GeoDataFrameDriver interface.

        Parameters
        ----------
        path : Path | str
            Destination path or URI where the GeoDataFrame would be written.
        data : gpd.GeoDataFrame
            The GeoDataFrame to write.
        write_kwargs : dict[str, Any], optional
            Additional keyword arguments that would be passed to the underlying write
            function. Default is None.

        Returns
        -------
        Path
            The path where the GeoDataFrame would be written.

        Raises
        ------
        NotImplementedError
            Always raised, as writing is not supported for this driver.
        """
        raise NotImplementedError(
            f"Writing using driver '{self.name}' is not supported."
        )
