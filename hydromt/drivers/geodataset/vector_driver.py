"""RasterDatasetDriver for zarr data."""

from copy import copy
from logging import Logger, getLogger
from typing import Callable, List, Optional

from xarray import Dataset

from hydromt._typing.error import NoDataStrategy
from hydromt._typing.type_def import Bbox, Geom, GeomBuffer, Predicate, TimeRange
from hydromt._utils.unused_kwargs import warn_on_unused_kwargs
from hydromt.drivers.geodataset.geodataset_driver import GeoDatasetDriver
from hydromt.drivers.preprocessing import PREPROCESSORS
from hydromt.io import open_geodataset

logger = getLogger(__name__)


class GeoDatasetVectorDriver(GeoDatasetDriver):
    """VectorGeodatasetDriver for vector data."""

    name = "vector"

    def read_data(
        self,
        uris: List[str],
        *,
        bbox: Optional[Bbox] = None,
        geom: Optional[Geom] = None,
        buffer: GeomBuffer = 0,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        single_var_as_array: bool = True,
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
    ) -> Dataset:
        """
        Read zarr data to an xarray DataSet.

        Args:
        """
        warn_on_unused_kwargs(
            self.__class__.__name__,
            {
                "buffer": buffer,
                "predicate": predicate,
                "variables": variables,
                "single_var_as_array": single_var_as_array,
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

        return open_geodataset(
            fn_locs=uri, bbox=bbox, geom=geom, logger=logger, **options
        )

    def write(self):
        """Not implemented."""
        raise NotImplementedError("GeodatasetVectorDriver does not support writing. ")
