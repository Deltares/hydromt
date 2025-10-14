"""DatasetDriver for zarr data."""

import logging
from functools import partial
from os.path import splitext
from typing import Any, Callable, ClassVar

import xarray as xr
from pydantic import Field

from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers.base_driver import (
    DRIVER_OPTIONS_DESCRIPTION,
    DriverOptions,
)
from hydromt.data_catalog.drivers.dataset.dataset_driver import DatasetDriver
from hydromt.data_catalog.drivers.preprocessing import get_preprocessor
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.typing import (
    SourceMetadata,
    StrPath,
    TimeRange,
)

logger = logging.getLogger(__name__)


class DatasetXarrayOptions(DriverOptions):
    """Options for DatasetXarrayDriver."""

    preprocess: str | None = Field(
        default=None,
        description="Name of preprocessor to apply before merging datasets. Available preprocessors include: round_latlon, to_datetimeindex, remove_duplicates, harmonise_dims. See their docstrings for details.",
    )
    ext_override: str | None = Field(
        default=None,
        description="Override the file extension check and try to read all files as the given extension. Useful when reading zarr files without the .zarr extension.",
    )

    def get_preprocessor(self) -> Callable | None:
        """Get the preprocessor function."""
        if self.preprocess is None:
            return None
        return get_preprocessor(self.preprocess)

    def get_ext_override(self, uris: list[str]) -> str:
        """Get the extension override."""
        if not self.ext_override:
            return splitext(uris[0])[-1]
        return self.ext_override


class DatasetXarrayDriver(DatasetDriver):
    """
    Driver for Dataset using xarray: ``dataset_xarray``.

    Supports reading and writing zarr and netcdf files using xarray.
    zarr files will be read using `xr.open_zarr` and netcdf files using
    `xr.open_mfdataset`.

    """

    name: ClassVar[str] = "dataset_xarray"
    supports_writing: ClassVar[bool] = True
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".zarr", ".nc", ".netcdf"}

    options: DatasetXarrayOptions = Field(
        default_factory=DatasetXarrayOptions, description=DRIVER_OPTIONS_DESCRIPTION
    )

    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        kwargs_for_open: dict[str, Any] | None = None,
        metadata: SourceMetadata | None = None,
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
                "time_range": time_range,
                "variables": variables,
                "metadata": metadata,
            },
        )
        preprocessor: Callable | None = self.options.get_preprocessor()
        first_ext = self.options.get_ext_override(uris)
        kwargs_for_open = kwargs_for_open or {}
        kwargs = self.options.get_kwargs() | kwargs_for_open
        if first_ext == ".zarr":
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
        elif first_ext in [".nc", ".netcdf"]:
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
