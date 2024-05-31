"""RasterDatasetDriver for zarr data."""

from copy import copy
from functools import partial
from logging import Logger, getLogger
from os.path import splitext
from typing import Callable, List, Optional

import xarray as xr

from hydromt._typing import (
    Geom,
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
    ZoomLevel,
)
from hydromt._typing.error import NoDataStrategy
from hydromt._utils.unused_kwargs import warn_on_unused_kwargs
from hydromt.data_catalog.drivers.preprocessing import PREPROCESSORS
from hydromt.data_catalog.drivers.raster.raster_dataset_driver import (
    RasterDatasetDriver,
)

logger: Logger = getLogger(__name__)


class RasterDatasetXarrayDriver(RasterDatasetDriver):
    """RasterDatasetXarrayDriver."""

    name = "raster_xarray"
    supports_writing = True

    def read_data(
        self,
        uris: List[str],
        *,
        logger: Logger = logger,
        mask: Optional[Geom] = None,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[ZoomLevel] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
    ) -> xr.Dataset:
        """
        Read zarr data to an xarray DataSet.

        Args:
        """
        warn_on_unused_kwargs(
            self.__class__.__name__,
            {
                "mask": mask,
                "time_range": time_range,
                "variables": variables,
                "zoom_level": zoom_level,
                "metadata": metadata,
            },
            logger,
        )
        options = copy(self.options)
        preprocessor: Optional[Callable] = None
        preprocessor_name: Optional[str] = options.pop("preprocess", None)
        if preprocessor_name:
            preprocessor = PREPROCESSORS.get(preprocessor_name)
            if not preprocessor:
                raise ValueError(f"unknown preprocessor: '{preprocessor_name}'")

        first_ext = splitext(uris[0])[-1]
        if first_ext == ".zarr":
            opn: Callable = partial(xr.open_zarr, **options)
            datasets = []
            for _uri in uris:
                ext = splitext(_uri)[-1]
                if ext != first_ext:
                    logger.warning(f"Reading zarr and {_uri} was not, skipping...")
                    continue
                else:
                    datasets.append(
                        preprocessor(opn(_uri)) if preprocessor else opn(_uri)
                    )

            return xr.merge(datasets)
        elif first_ext in [".nc", ".netcdf"]:
            filtered_uris = []
            for _uri in uris:
                ext = splitext(_uri)[-1]
                if ext != first_ext:
                    logger.warning(f"Reading netcdf and {_uri} was not, skipping...")
                    continue
                else:
                    filtered_uris.append(_uri)

            return xr.open_mfdataset(
                filtered_uris, decode_coords="all", preprocess=preprocessor, **options
            )
        else:
            raise ValueError(
                f"Unknown extension for RasterDatasetXarrayDriver: {first_ext} "
            )

    def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
        """
        Write the RasterDataset to a local file using zarr.

        args:
        """
        ext = splitext(path)[-1]
        if ext == ".zarr":
            ds.to_zarr(path, **kwargs)
        elif ext in [".nc", ".netcdf"]:
            ds.to_netcdf(path, **kwargs)
        else:
            raise ValueError(f"Unknown extension for RasterDatasetXarrayDriver: {ext} ")
