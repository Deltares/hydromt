"""GeoDatasetVectorDriver class for reading vector data from table like files such as csv or parquet."""

import logging
from pathlib import Path
from typing import Any, ClassVar

import xarray as xr
from pydantic import Field

from hydromt.data_catalog.drivers.base_driver import (
    DRIVER_OPTIONS_DESCRIPTION,
)
from hydromt.data_catalog.drivers.geodataset.geodataset_driver import (
    GeoDatasetDriver,
    GeoDatasetOptions,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.readers import open_geodataset
from hydromt.typing import CRS, Geom, Predicate, SourceMetadata

logger = logging.getLogger(__name__)


class GeoDatasetVectorDriver(GeoDatasetDriver):
    """
    Driver for GeoDataset using hydromt vector: ``geodataframe_vector``.

    Supports reading geodataset from a combination of a geometry file and optionally an
    external data file. The geometry file can be any file supported by
    `geopandas.read_file`, such as shapefile, geojson, geopackage, or a tabular file
    like csv or parquet for points (containing latitude and longitude columns). The
    external data file can be a netcdf file or any tabular file like csv or parquet.
    The geometry file and the external data file can be linked using a common column
    (e.g. an ID column).

    """

    name: ClassVar[str] = "geodataset_vector"
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {
        ".csv",
        ".parquet",
        ".xlsx",
        ".xls",
        ".xy",
        ".gpkg",
        ".shp",
        ".geojson",
        ".fgb",
    }

    options: GeoDatasetOptions = Field(
        default_factory=GeoDatasetOptions, description=DRIVER_OPTIONS_DESCRIPTION
    )

    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        mask: Geom | None = None,
        predicate: Predicate = "intersects",
        metadata: SourceMetadata | None = None,
    ) -> xr.Dataset:
        """
        Read tabular or vector dataset files (e.g., CSV, Parquet) into an xarray Dataset.

        Parameters
        ----------
        uris : list[str]
            List of URIs to read data from.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle missing data. Default is NoDataStrategy.RAISE.
        mask : Geom | None, optional
            Optional spatial mask to clip the dataset.
        predicate : Predicate, optional
            Spatial predicate for filtering geometries. Default is "intersects".
        metadata : SourceMetadata | None, optional
            Optional metadata object to attach to the loaded dataset.

        Returns
        -------
        xr.Dataset
            The dataset read from the source.
        """
        if len(uris) > 1:
            raise ValueError(
                "GeodatasetVectorDriver only supports reading from one URI per source"
            )
        else:
            uri = uris[0]

        preprocessor = self.options.get_preprocessor()
        crs: CRS | None = metadata.crs if metadata else None
        data = open_geodataset(
            loc_path=uri,
            geom=mask,
            crs=crs,
            predicate=predicate,
            **self.options.get_kwargs(),
        )
        out = preprocessor(data)

        if isinstance(out, xr.DataArray):
            if out.size == 0:
                exec_nodata_strat(
                    f"No data from {self.name} driver for file uris: {', '.join(uris)}.",
                    strategy=handle_nodata,
                )
                return None  # handle_nodata == ignore
            return out.to_dataset()
        else:
            for variable in out.data_vars:
                if out[variable].size == 0:
                    exec_nodata_strat(
                        f"No data from {self.name} driver for file uris: {', '.join(uris)}.",
                        strategy=handle_nodata,
                    )
                    return None  # handle_nodata == ignore
            return out

    def write(
        self,
        path: Path | str,
        data: xr.Dataset,
        *,
        write_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write a GeoDataset to disk.

        Writing is not supported for this driver, as vector and tabular sources such as
        CSV or Parquet are read-only in this context. This method exists for interface
        consistency with GeoDatasetDriver.

        Parameters
        ----------
        path : Path | str
            Destination path where the dataset would be written.
        data : xr.Dataset
            The dataset to write.
        write_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments that would be passed to the underlying write
            function. Default is None.

        Raises
        ------
        NotImplementedError
            Always raised, as writing is not supported for this driver.
        """
        raise NotImplementedError("GeodatasetVectorDriver does not support writing. ")
