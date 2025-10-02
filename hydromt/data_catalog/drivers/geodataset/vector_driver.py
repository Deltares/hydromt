"""GeoDatasetVectorDriver class for reading vector data from table like files such as csv or parquet."""

from copy import copy
from logging import getLogger
from typing import Callable, ClassVar, List, Optional

import xarray as xr

from hydromt._typing import CRS, SourceMetadata
from hydromt._typing.type_def import Geom, Predicate, StrPath, TimeRange
from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers.geodataset.geodataset_driver import GeoDatasetDriver
from hydromt.data_catalog.drivers.preprocessing import PREPROCESSORS
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.io import open_geodataset

logger = getLogger(__name__)


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

    Driver **options** include:

    * preprocess: Optional[str], name of preprocessor to apply on geodataset after
      reading. Available preprocessors include: round_latlon, to_datetimeindex,
      remove_duplicates, harmonise_dims. See their docstrings for details.
    * Any other option supported by `hydromt.io.open_geodataset`.

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

    def read(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
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

        options = copy(self.options)
        preprocessor: Optional[Callable] = None
        preprocessor_name: Optional[str] = options.pop("preprocess", None)
        if preprocessor_name:
            preprocessor = PREPROCESSORS.get(preprocessor_name)
            if not preprocessor:
                raise ValueError(f"unknown preprocessor: '{preprocessor_name}'")

        crs: Optional[CRS] = metadata.crs if metadata else None
        data = open_geodataset(
            loc_path=uri, geom=mask, crs=crs, predicate=predicate, **options
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
