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
    Zoom,
)
from hydromt._typing.error import NoDataStrategy, exec_nodata_strat
from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers._preprocessing import PREPROCESSORS
from hydromt.data_catalog.drivers.raster.raster_dataset_driver import (
    RasterDatasetDriver,
)

logger: Logger = getLogger(__name__)

_ZARR_EXT = ".zarr"


class RasterDatasetXarrayDriver(RasterDatasetDriver):
    """RasterDatasetXarrayDriver."""

    name = "raster_xarray"
    supports_writing = True

    def read(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        zoom: Optional[Zoom] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> xr.Dataset:
        """
        Read zarr data to an xarray DataSet.

        Args:
        """
        _warn_on_unused_kwargs(
            self.__class__.__name__,
            {
                "mask": mask,
                "time_range": time_range,
                "variables": variables,
                "zoom": zoom,
                "metadata": metadata,
            },
        )
        options = copy(self.options)
        preprocessor: Optional[Callable] = None
        preprocessor_name: Optional[str] = options.pop("preprocess", None)
        if preprocessor_name:
            preprocessor = PREPROCESSORS.get(preprocessor_name)
            if not preprocessor:
                raise ValueError(f"unknown preprocessor: '{preprocessor_name}'")

        ext_override: Optional[str] = options.pop("ext_override", None)
        if ext_override is not None:
            first_ext: str = ext_override
        else:
            first_ext: str = splitext(uris[0])[-1]

        if first_ext == _ZARR_EXT:
            opn: Callable = partial(xr.open_zarr, **options)
            datasets = []
            for _uri in uris:
                ext = splitext(_uri)[-1]
                if ext != first_ext and not ext_override:
                    logger.warning(f"Reading zarr and {_uri} was not, skipping...")
                else:
                    datasets.append(
                        preprocessor(opn(_uri)) if preprocessor else opn(_uri)
                    )

            ds: xr.Dataset = xr.merge(datasets)
        elif first_ext in [".nc", ".netcdf"]:
            filtered_uris = []
            for _uri in uris:
                ext = splitext(_uri)[-1]
                if ext != first_ext:
                    logger.warning(f"Reading netcdf and {_uri} was not, skipping...")
                else:
                    filtered_uris.append(_uri)

            ds: xr.Dataset = xr.open_mfdataset(
                filtered_uris, decode_coords="all", preprocess=preprocessor, **options
            )
        else:
            raise ValueError(
                f"Unknown extension for RasterDatasetXarrayDriver: {first_ext} "
            )
        for variable in ds.data_vars:
            if ds[variable].size == 0:
                exec_nodata_strat(
                    f"No data from driver: '{self.name}' for variable: '{variable}'",
                    strategy=handle_nodata,
                )
        return ds

    def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> str:
        """
        Write the RasterDataset to a local file using zarr.

        args:

        returns: str with written uri
        """
        no_ext, ext = splitext(path)
        # set filepath if incompat
        if ext not in {_ZARR_EXT, ".nc", ".netcdf"}:
            logger.warning(
                f"Unknown extension for RasterDatasetXarrayDriver: {ext},"
                "switching to zarr"
            )
            path = no_ext + _ZARR_EXT
            ext = _ZARR_EXT
        if ext == _ZARR_EXT:
            ds.to_zarr(path, mode="w", **kwargs)
        else:
            ds.to_netcdf(path, **kwargs)

        return str(path)
