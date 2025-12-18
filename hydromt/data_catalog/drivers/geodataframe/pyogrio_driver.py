"""Driver to read geodataframes using Pyogrio."""

import logging
from os.path import splitext
from pathlib import Path
from typing import Any, ClassVar

import geopandas as gpd
import pandas as pd
from pyogrio import read_dataframe, read_info, write_dataframe
from pyproj import CRS

from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers.geodataframe.geodataframe_driver import (
    GeoDataFrameDriver,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.typing import Bbox, Geom, SourceMetadata

logger = logging.getLogger(__name__)


class PyogrioDriver(GeoDataFrameDriver):
    """
    Driver for GeoDataFrame using the pyogrio library: ``pyogrio``.

    Supports reading and writing files supported by the OGR library,
    including geopackage, shapefile, geojson and flatgeobuf.
    """

    name: ClassVar[str] = "pyogrio"
    supports_writing: ClassVar[bool] = True
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".gpkg", ".shp", ".geojson", ".fgb"}

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
        Read geospatial data using the pyogrio library into a GeoDataFrame.

        Supports formats such as GeoPackage, Shapefile, GeoJSON, and FlatGeobuf.
        Optionally applies spatial filtering through a bounding box derived
        from a provided mask.

        Parameters
        ----------
        uris : list[str]
            List of URIs to read data from. Only one file is supported per read operation.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle missing or empty data. Default is NoDataStrategy.RAISE.
        metadata : SourceMetadata | None, optional
            Optional metadata object describing the dataset source (e.g. CRS).
        mask : Any, optional
            Optional geometry or GeoDataFrame used to spatially filter the data
            while reading.
        variables : str | list[str] | None, optional
            Optional list of columns to load from the dataset.

        Returns
        -------
        gpd.GeoDataFrame
            The loaded geospatial data.

        Raises
        ------
        ValueError
            If multiple URIs are provided.
        IOError
            If the source file contains no geometry column.

        Warning
        -------
        The `metadata` parameter is not used directly in this driver, but is included
        for consistency with the GeoDataFrameDriver interface.
        """
        _warn_on_unused_kwargs(self.__class__.__name__, {"metadata": metadata})

        if len(uris) > 1:
            raise ValueError(
                "DataFrame: Reading multiple files with the "
                f"{self.__class__.__name__} driver is not supported."
            )
        _uri = uris[0]
        if mask is not None:
            bbox = _bbox_from_file_and_mask(
                _uri, mask=mask, **self.options.get_kwargs()
            )
        else:
            bbox = None
        gdf: pd.DataFrame | gpd.GeoDataFrame = read_dataframe(
            _uri, bbox=bbox, columns=variables, **self.options.get_kwargs()
        )
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise IOError(f"DataFrame from uri: '{_uri}' contains no geometry column.")

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
        Write a GeoDataFrame to disk using the pyogrio library.

        Supports writing to vector formats supported by the OGR library, including
        GeoPackage (`.gpkg`), Shapefile (`.shp`), GeoJSON (`.geojson`), and FlatGeobuf (`.fgb`).
        The file format is inferred from the file extension. If the extension is unsupported,
        it falls back to FlatGeobuf (`.fgb`).

        Parameters
        ----------
        path : Path | str
            Destination path or URI where the GeoDataFrame will be written.
            Supported extensions are `.gpkg`, `.shp`, `.geojson`, and `.fgb`.
        data : gpd.GeoDataFrame
            The GeoDataFrame to write.
        write_kwargs : dict[str, Any], optional
            Additional keyword arguments passed to `pyogrio.write_dataframe`. Default is None.

        Returns
        -------
        Path
            The path where the GeoDataFrame was written.

        Raises
        ------
        ValueError
            If the file extension cannot be determined or writing fails.
        """
        no_ext, ext = splitext(path)
        write_kwargs = write_kwargs or {}
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning(
                f"driver {self.name} has no support for extension {ext}"
                "switching to .fgb."
            )
            path = no_ext + ".fgb"

        write_dataframe(data, path, **write_kwargs)

        return Path(path)


def _bbox_from_file_and_mask(
    uri: str,
    mask: Geom,
    **kwargs,
) -> Bbox | None:
    """Create a bbox from the file metadata and mask given.

    Pyogrio's mask or bbox arguments require a mask or bbox in the same CRS as the data.
    This function takes the mask filter and crs of the input data
    and returns a bbox in the same crs as the data based on the input filters.

    Parameters
    ----------
    uri: str,
        URI of the data.
    mask: GeoDataFrame | GeoSeries | BaseGeometry
        mask to filter the data while reading.
    """
    source_crs = None
    if source_crs_str := read_info(uri, **kwargs).get("crs"):
        source_crs = CRS.from_user_input(source_crs_str)

    if not source_crs:
        logger.warning(
            f"Reading from uri: '{uri}' without CRS definition. Filtering with crs:"
            f" {mask.crs}, cannot compare crs."
        )
    elif mask.crs != source_crs:
        mask = mask.to_crs(source_crs)

    return tuple(mask.total_bounds)
