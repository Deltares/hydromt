"""Driver for reading in GeoDataFrames from tabular formats."""
from logging import Logger, getLogger
from typing import List, Optional, Set

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from hydromt._typing import Geom
from hydromt._typing.error import NoDataStrategy
from hydromt._typing.metadata import SourceMetadata
from hydromt._utils.unused_kwargs import warn_on_unused_kwargs
from hydromt.data_catalog.drivers.geodataframe.geodataframe_driver import (
    GeoDataFrameDriver,
)

logger: Logger = getLogger(__name__)

# possible labels for the x and y dimensions
X_DIM_LABELS = ("x", "longitude", "lon", "long")
Y_DIM_LABELS = ("y", "latitude", "lat")


class GeoDataFrameTableDriver(GeoDataFrameDriver):
    """Driver for reading in GeoDataFrames from tabular formats."""

    name = "geodataframe_table"

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
        """Read tabular data using a combination of the pandas and geopandas libraries."""
        if not metadata:
            metadata = SourceMetadata()
        warn_on_unused_kwargs(
            self.__class__.__name__,
            {"mask": mask, "predicate": predicate, "variables": variables},
            logger=logger,
        )
        if len(uris) > 1:
            raise ValueError(
                "DataFrame: Reading multiple files with the "
                f"{self.__class__.__name__} driver is not supported."
            )

        return open_vector_from_table(
            uri=uris[0],
            x_dim=self.options.get("x_dim"),
            y_dim=self.options.get("y_dim"),
            crs=metadata.crs,
        )


def open_vector_from_table(
    uri: str,
    x_dim: Optional[str] = None,
    y_dim: Optional[str] = None,
    crs: Optional[CRS] = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    r"""Read point geometry files from csv, parquet, xy or excel table files.

    Parameters
    ----------
    x_dim, y_dim: str
        Name of x, y column. By default the x-column header should be one of
        ['x', 'longitude', 'lon', 'long'], and y-column header one of
        ['y', 'latitude', 'lat']. For xy files, which don't have a header,
        the first column is interpreted as x and the second as y column.
    crs: int, dict, or str, optional
        Coordinate reference system, accepts EPSG codes (int or str), proj (str or dict)
        or wkt (str)
    **kwargs
        Additional keyword arguments that are passed to the underlying drivers.

    Returns
    -------
    gdf: geopandas.GeoDataFrame
        Parsed and filtered point geometries
    """
    ext: str = uri.split(".")[-1].lower()
    if "index_col" not in kwargs and ext != "parquet":
        kwargs.update(index_col=0)
    if ext == "csv":
        df = pd.read_csv(uri, **kwargs)
    elif ext == "parquet":
        df = pd.read_parquet(uri, **kwargs)
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(uri, engine="openpyxl", **kwargs)
    else:
        raise ValueError(
            f"Extension '{ext}' not compatible with geodataframe table driver"
        )

    # Make columns case insensitive
    columns: Set[str] = set(map(lambda col: col.lower(), df.columns))
    columns.update(set(map(lambda col: col.upper(), df.columns)))

    if x_dim is None:
        for dim in X_DIM_LABELS:
            if dim in columns:
                x_dim = dim
                break
    if x_dim is None or x_dim not in columns:
        raise ValueError(f'x dimension "{x_dim}" not found in columns: {columns}.')
    if y_dim is None:
        for dim in Y_DIM_LABELS:
            if dim in df.columns:
                y_dim = dim
                break
    if y_dim is None or y_dim not in columns:
        raise ValueError(f'y dimension "{y_dim}" not found in columns: {columns}.')
    points = gpd.points_from_xy(df[x_dim], df[y_dim])
    gdf = gpd.GeoDataFrame(df.drop(columns=[x_dim, y_dim]), geometry=points, crs=crs)
    return gdf
