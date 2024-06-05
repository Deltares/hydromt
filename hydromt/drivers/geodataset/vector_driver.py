"""GeoDatasetVectorDriver class for reading vector data from table like files such as csv or parquet."""

from copy import copy
from logging import Logger, getLogger
from typing import Callable, ClassVar, List, Optional

import xarray as xr

from hydromt._typing import CRS, SourceMetadata
from hydromt._typing.error import NoDataStrategy, exec_nodata_strat
from hydromt._typing.type_def import Geom, Predicate, StrPath, TimeRange
from hydromt._utils.unused_kwargs import warn_on_unused_kwargs
from hydromt.drivers.geodataset.geodataset_driver import GeoDatasetDriver
from hydromt.drivers.preprocessing import PREPROCESSORS
from hydromt.io import open_geodataset

logger = getLogger(__name__)


class GeoDatasetVectorDriver(GeoDatasetDriver):
    """VectorGeodatasetDriver for vector data."""

    name: ClassVar[str] = "geodataset_vector"

    def read_data(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        single_var_as_array: bool = True,
        metadata: Optional[SourceMetadata] = None,
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
    ) -> xr.Dataset:
        """
        Read tabular datafiles like csv or parquet into to an xarray DataSet.

        Args:
        """
        warn_on_unused_kwargs(
            self.__class__.__name__,
            {
                "predicate": predicate,
                "variables": variables,
                "time_range": time_range,
                "single_var_as_array": single_var_as_array,
                "metadata": metadata,
            },
            logger,
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
            fn_locs=uri, geom=mask, logger=logger, crs=crs, **options
        )

        if preprocessor is None:
            out = data
        else:
            out = preprocessor(data)

        if isinstance(out, xr.DataArray):
            if out.size == 0:
                exec_nodata_strat(
                    f"No data from driver {self}'.",
                    strategy=handle_nodata,
                    logger=logger,
                )
                return out.to_dataset()
        else:
            for variable in out.data_vars:
                if out[variable].size == 0:
                    exec_nodata_strat(
                        f"No data from driver {self}' for variable {variable}.",
                        strategy=handle_nodata,
                        logger=logger,
                    )
            return out

    def write(
        self,
        path: StrPath,
        ds: xr.Dataset,
        **kwargs,
    ) -> xr.Dataset:
        """Not implemented."""
        raise NotImplementedError("GeodatasetVectorDriver does not support writing. ")
