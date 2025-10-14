"""GeoDatasetDriver for zarr data."""

import logging
from functools import partial
from os.path import splitext
from typing import Any, Callable, ClassVar

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
from hydromt.typing import (
    Geom,
    SourceMetadata,
    StrPath,
    TimeRange,
)
from hydromt.typing.type_def import Predicate

logger = logging.getLogger(__name__)

_ZARR_EXT = ".zarr"
_NETCDF_EXT = [".nc", ".netcdf"]


class GeoDatasetXarrayDriver(GeoDatasetDriver):
    """
    Driver for GeoDataset using the xarray library: ``geodataset_xarray``.

    Supports reading and writing zarr and netcdf files using xarray.
    zarr files will be read using `xr.open_zarr` and netcdf files using
    `xr.open_mfdataset`.
    """

    name: ClassVar[str] = "geodataset_xarray"
    supports_writing = True
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {_ZARR_EXT, *_NETCDF_EXT}
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
        metadata: SourceMetadata | None = None,
        predicate: Predicate = "intersects",
        time_range: TimeRange | None = None,
        variables: str | list[str] | None = None,
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

        preprocessor = self.options.get_preprocessor()
        first_ext = splitext(uris[0])[-1]
        kwargs_for_open = kwargs_for_open or {}
        kwargs = self.options.get_kwargs() | kwargs_for_open
        if first_ext == _ZARR_EXT:
            opn: Callable = partial(xr.open_zarr, **kwargs)
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
        elif first_ext in _NETCDF_EXT:
            filtered_uris = []
            for _uri in uris:
                ext = splitext(_uri)[-1]
                if ext != first_ext:
                    logger.warning(f"Reading netcdf and {_uri} was not, skipping...")
                else:
                    filtered_uris.append(_uri)

            ds: xr.Dataset = xr.open_mfdataset(
                filtered_uris, decode_coords="all", preprocess=preprocessor, **kwargs
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
        if ext == _ZARR_EXT:
            ds.vector.to_zarr(path, **kwargs)
        elif ext in _NETCDF_EXT:
            ds.vector.to_netcdf(path, **kwargs)
        else:
            raise ValueError(f"Unknown extension for GeoDatasetXarrayDriver: {ext} ")

        return path
