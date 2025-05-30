"""GeoDatasetDriver for zarr data."""

from copy import copy
from functools import partial
from logging import Logger, getLogger
from os.path import splitext
from typing import Callable, ClassVar, List, Optional

import xarray as xr

from hydromt._typing import (
    Geom,
    SourceMetadata,
    StrPath,
    TimeRange,
)
from hydromt._typing.error import NoDataStrategy, exec_nodata_strat
from hydromt._typing.type_def import Predicate
from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers._preprocessing import PREPROCESSORS
from hydromt.data_catalog.drivers.geodataset.geodataset_driver import GeoDatasetDriver

logger: Logger = getLogger(__name__)


class GeoDatasetXarrayDriver(GeoDatasetDriver):
    """GeoDatasetXarrayDriver."""

    name: ClassVar[str] = "geodataset_xarray"
    supports_writing = True

    def read(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        metadata: Optional[SourceMetadata] = None,
        predicate: Predicate = "intersects",
        time_range: Optional[TimeRange] = None,
        variables: Optional[List[str]] = None,
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
                "predicate": predicate,
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
                f"Unknown extention for GeoDatasetXarrayDriver: {first_ext} "
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
        Write the GeoDataset to a local file using zarr.

        args:
        """
        ext = splitext(path)[-1]
        if ext == ".zarr":
            ds.vector.to_zarr(path, **kwargs)
        elif ext in [".nc", ".netcdf"]:
            ds.vector.to_netcdf(path, **kwargs)
        else:
            raise ValueError(f"Unknown extension for GeoDatasetXarrayDriver: {ext} ")

        return path
