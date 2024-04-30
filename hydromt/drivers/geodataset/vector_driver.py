"""RasterDatasetDriver for zarr data."""

from copy import copy
from functools import partial
from logging import Logger
from typing import Callable, List, Optional

import xarray as xr

from hydromt._typing import StrPath
from hydromt._typing.error import NoDataStrategy
from hydromt._utils.unused_kwargs import warn_on_unused_kwargs
from hydromt.drivers.geodataset.geodataset_driver import GeoDatasetDriver
from hydromt.drivers.preprocessing import PREPROCESSORS


class GeoDatasetVectorDriver(GeoDatasetDriver):
    """VectorGeodatasetDriver for vector data."""

    name = "geods_vector"

    def read_data(
        self,
        uris: List[str],
        *,
        logger: Optional[Logger] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
    ) -> xr.Dataset:
        """
        Read zarr data to an xarray DataSet.

        Args:
        """
        warn_on_unused_kwargs(
            self.__class__.__name__,
            {"mask": mask, "time_range": time_range, "zoom_level": zoom_level},
            logger,
        )
        options = copy(self.options)
        preprocessor: Optional[Callable] = None
        preprocessor_name: Optional[str] = options.pop("preprocess", None)
        if preprocessor_name:
            preprocessor = PREPROCESSORS.get(preprocessor_name)
            if not preprocessor:
                raise ValueError(f"unknown preprocessor: '{preprocessor_name}'")

        opn: Callable = partial(xr.open_zarr, **options)

        return xr.merge(
            [preprocessor(opn(_uri)) if preprocessor else opn(_uri) for _uri in uris]
        )

    def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
        """
        Write the RasterDataset to a local file using zarr.

        args:
        """
        ds.to_zarr(path, **kwargs)
