"""Driver for reading in GeoDataFrames from tabular formats."""

import logging
from typing import Any, ClassVar

import geopandas as gpd
from pydantic import Field

from hydromt.data_catalog.drivers.base_driver import (
    DRIVER_OPTIONS_DESCRIPTION,
    DriverOptions,
)
from hydromt.data_catalog.drivers.geodataframe.geodataframe_driver import (
    GeoDataFrameDriver,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.io.readers import open_vector_from_table
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
        kwargs_for_open: dict[str, Any] | None = None,
        metadata: SourceMetadata | None = None,
        mask: Any = None,
        predicate: str = "intersects",
        variables: str | list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """Read tabular data using a combination of the pandas and geopandas libraries."""
        if not metadata:
            metadata = SourceMetadata()
        if len(uris) > 1:
            raise ValueError(
                "DataFrame: Reading multiple files with the "
                f"{self.__class__.__name__} driver is not supported."
            )

        _uri: str = uris[0]
        kwargs_for_open = kwargs_for_open or {}
        kwargs = self.options.get_kwargs() | kwargs_for_open
        gdf = open_vector_from_table(
            path=_uri,
            x_dim=self.options.x_dim,
            y_dim=self.options.y_dim,
            crs=metadata.crs,
            **kwargs,
        )
        if gdf.index.size == 0:
            exec_nodata_strat(
                f"No data from driver {self}'.",
                strategy=handle_nodata,
            )
        return gdf
