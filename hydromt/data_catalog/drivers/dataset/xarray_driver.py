"""DatasetDriver for zarr data."""

from copy import copy
from functools import partial
from logging import Logger, getLogger
from os.path import splitext
from typing import Callable, ClassVar, List, Optional

import xarray as xr

from hydromt._typing import (
    SourceMetadata,
    StrPath,
    TimeRange,
)
from hydromt._typing.error import NoDataStrategy, exec_nodata_strat
from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers._preprocessing import PREPROCESSORS
from hydromt.data_catalog.drivers.dataset.dataset_driver import DatasetDriver

logger: Logger = getLogger(__name__)


class DatasetXarrayDriver(DatasetDriver):
    """DatasetXarrayDriver."""

    name: ClassVar[str] = "dataset_xarray"
    supports_writing = True

    def read(
        self,
        uris: List[str],
        *,
        metadata: Optional[SourceMetadata] = None,
        time_range: Optional[TimeRange] = None,
        variables: Optional[List[str]] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
    ) -> xr.Dataset:
        """
        Read zarr data to an xarray DataSet.

        Args:
        """
        _warn_on_unused_kwargs(
            self.__class__.__name__,
            {
                "time_range": time_range,
                "variables": variables,
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

            ds: xr.Dataset = xr.merge(datasets)
        elif first_ext in [".nc", ".netcdf"]:
            filtered_uris = []
            for _uri in uris:
                ext = splitext(_uri)[-1]
                if ext != first_ext:
                    logger.warning(f"Reading netcdf and {_uri} was not, skipping...")
                    continue
                else:
                    filtered_uris.append(_uri)

            ds: xr.Dataset = xr.open_mfdataset(
                filtered_uris, decode_coords="all", preprocess=preprocessor, **options
            )
        else:
            raise ValueError(f"Unknown extension for DatasetXarrayDriver: {first_ext} ")
        for variable in ds.data_vars:
            if ds[variable].size == 0:
                exec_nodata_strat(
                    f"No data from driver: '{self.name}' for variable: '{variable}'",
                    strategy=handle_nodata,
                )
        return ds

    def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> str:
        """
        Write the Dataset to a local file using zarr.

        args:
        """
        ext = splitext(path)[-1]
        if ext == ".zarr":
            ds.to_zarr(path, **kwargs)
        elif ext in [".nc", ".netcdf"]:
            ds.to_netcdf(path, **kwargs)
        else:
            raise ValueError(f"Unknown extension for DatasetXarrayDriver: {ext} ")

        return str(path)
