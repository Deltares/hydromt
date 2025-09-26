"""Driver for reading in GeoDataFrames from tabular formats."""

from copy import deepcopy
from logging import Logger, getLogger
from typing import ClassVar, List, Optional

import geopandas as gpd

from hydromt._typing.metadata import SourceMetadata
from hydromt.data_catalog.drivers.geodataframe.geodataframe_driver import (
    GeoDataFrameDriver,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.io.readers import open_vector_from_table

logger: Logger = getLogger(__name__)

# possible labels for the x and y dimensions
X_DIM_LABELS = ("x", "longitude", "lon", "long")
Y_DIM_LABELS = ("y", "latitude", "lat")


class GeoDataFrameTableDriver(GeoDataFrameDriver):
    """
    Driver for GeoDataFrame from tabular formats: ``geodataframe_table``.

    Supports reading point geometries from csv, excel (xls, xlsx), and parquet files
    using a combination of the pandas and geopandas libraries.

    Driver **options** include:

    * x_dim: Optional[str], name of the column containing the x coordinate. Not needed
      if one of the default names is used ('x', 'longitude', 'lon', 'long').
    * y_dim: Optional[str], name of the column containing the y coordinate. Not needed
      if one of the default names is used ('y', 'latitude', 'lat').
    * Any other option supported by the underlying pandas read functions,
      e.g. `pd.read_csv`, `pd.read_excel`, `pd.read_parquet`.

    """

    name: ClassVar[str] = "geodataframe_table"
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".csv", ".xlsx", ".xls", ".parquet"}

    def read(
        self,
        uris: List[str],
        *,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
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

        options = deepcopy(self.options)
        x_dim = options.pop("x_dim", None)
        y_dim = options.pop("y_dim", None)
        gdf = open_vector_from_table(
            path=_uri,
            x_dim=x_dim,
            y_dim=y_dim,
            crs=metadata.crs,
            **options,
        )
        if gdf.index.size == 0:
            exec_nodata_strat(
                f"No data from driver {self}'.",
                strategy=handle_nodata,
            )
        return gdf
