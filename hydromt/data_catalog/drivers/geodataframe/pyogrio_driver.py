"""Driver to read geodataframes using Pyogrio."""

import logging
from os.path import splitext
from pathlib import Path
from typing import Any, ClassVar

import geopandas as gpd
from pyogrio import write_dataframe

from hydromt.data_catalog.drivers.base_driver import resolve_filesystem
from hydromt.data_catalog.drivers.geodataframe.geodataframe_driver import (
    GeoDataFrameDriver,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.readers import open_vector
from hydromt.typing import SourceMetadata

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
    ) -> gpd.GeoDataFrame | None:
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
            Optional metadata object describing the dataset source. Its ``crs`` is
            used when the source file has no CRS of its own.
        mask : Any, optional
            Optional geometry or GeoDataFrame used to spatially filter the data
            while reading.
        variables : str | list[str] | None, optional
            Optional list of columns to load from the dataset.

        Returns
        -------
        gpd.GeoDataFrame | None
            The loaded geospatial data. Returns None if no data is available
            and the handle_nodata strategy is set to ignore.

        Raises
        ------
        ValueError
            If multiple URIs are provided.
        IOError
            If the source file contains no geometry column.
        """
        if len(uris) > 1:
            raise ValueError(
                "DataFrame: Reading multiple files with the "
                f"{self.__class__.__name__} driver is not supported."
            )

        # storage_options are handled once, here at the driver boundary.
        options = self.options.get_kwargs()
        fs = resolve_filesystem(self.filesystem, options)
        gdf = open_vector(
            uris[0],
            driver="pyogrio",
            crs=metadata.crs if metadata else None,
            geom=mask,
            columns=variables,
            filesystem=fs,
            **options,
        )
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise IOError(
                f"DataFrame from uri: '{uris[0]}' contains no geometry column."
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
