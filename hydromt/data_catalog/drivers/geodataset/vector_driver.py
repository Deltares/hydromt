"""GeoDatasetVectorDriver class for reading vector data from table like files such as csv or parquet."""

import logging
from typing import Any, ClassVar

import xarray as xr
from pydantic import Field

from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers.base_driver import (
    DRIVER_OPTIONS_DESCRIPTION,
)
from hydromt.data_catalog.drivers.geodataset.geodataset_driver import (
    GeoDatasetDriver,
    GeoDatasetOptions,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.io import open_geodataset
from hydromt.typing import CRS, Geom, Predicate, SourceMetadata, StrPath, TimeRange

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
        kwargs_for_open: dict[str, Any] | None = None,
        mask: Geom | None = None,
        predicate: Predicate = "intersects",
        variables: list[str] | None = None,
        time_range: TimeRange | None = None,
        metadata: SourceMetadata | None = None,
    ) -> xr.Dataset:
        """
        Read tabular datafiles like csv or parquet into to an xarray DataSet.

        Args:
        """
        _warn_on_unused_kwargs(
            self.__class__.__name__,
            {
                "variables": variables,
                "time_range": time_range,
                "metadata": metadata,
            },
        )
        # we want to maintain a list as argument to keep the interface compatible with other drivers.
        if len(uris) > 1:
            raise ValueError(
                "GeodatasetVectorDriver only supports reading from one URI per source"
            )
        else:
            uri = uris[0]

        preprocessor = self.options.get_preprocessor()
        kwargs_for_open = kwargs_for_open or {}
        kwargs = self.options.get_kwargs() | kwargs_for_open
        crs: CRS | None = metadata.crs if metadata else None
        data = open_geodataset(
            loc_path=uri, geom=mask, crs=crs, predicate=predicate, **kwargs
        )

        if preprocessor is None:
            out = data
        else:
            out = preprocessor(data)

        if isinstance(out, xr.DataArray):
            if out.size == 0:
                exec_nodata_strat(
                    f"No data from driver {self}'.", strategy=handle_nodata
                )
                return out.to_dataset()
        else:
            for variable in out.data_vars:
                if out[variable].size == 0:
                    exec_nodata_strat(
                        f"No data from driver {self}' for variable {variable}.",
                        strategy=handle_nodata,
                    )
            return out

    def write(
        self,
        path: StrPath,
        ds: xr.Dataset,
        **kwargs,
    ) -> str:
        """Not implemented."""
        raise NotImplementedError("GeodatasetVectorDriver does not support writing. ")
