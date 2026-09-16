"""Options for configuring xarray-based drivers in the data catalog."""

import enum
import logging
from os.path import splitext
from typing import Any

import xarray as xr
from dask import delayed
from dask.base import get_scheduler
from pydantic import Field

from hydromt.data_catalog.drivers.base_driver import DriverOptions
from hydromt.data_catalog.drivers.preprocessing import (
    Preprocessor,
    get_preprocessor,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.readers import open_mfdataset, open_zarrs
from hydromt.typing.fsspec_types import FSSpecFileSystem

logger = logging.getLogger(__name__)


class XarrayIOFormat(enum.Enum):
    """Enumeration of supported xarray IO formats, each associated with a set of file extensions."""

    ZARR = "zarr"
    NETCDF4 = "netcdf4"

    @property
    def extensions(self) -> set[str]:
        """Get the set of file extensions associated with this IO format."""
        if self == XarrayIOFormat.ZARR:
            return {".zarr"}
        elif self == XarrayIOFormat.NETCDF4:
            return {".nc", ".netcdf"}
        return set()


class XarrayDriverOptions(DriverOptions):
    """Options for configuring xarray-based drivers."""

    preprocess: str | None = Field(
        default=None,
        description="Name of preprocessor to apply before merging datasets. Available preprocessors include: round_latlon, to_datetimeindex, remove_duplicates, harmonise_dims. See their docstrings for details.",
    )
    ext_override: str | None = Field(
        default=None,
        description="Override the file extension check and try to read all files as the given extension. Useful when reading zarr files without the .zarr extension.",
    )

    def get_preprocessor(self) -> Preprocessor:
        """Get the preprocessor instance based on the configured preprocess string."""
        return get_preprocessor(self.preprocess)

    def get_reading_ext(self, uri: str) -> str:
        """Determine the file extension to use for reading, either from the override or from the URI."""
        return self.ext_override or splitext(uri)[-1]

    def get_io_format(self, uri: str) -> XarrayIOFormat | None:
        """Determine the xarray reading format based on the file extension or override.

        Parameters
        ----------
        uri : str
            The URI of the file to read, used to infer the format from the extension if no override is set.

        Returns
        -------
        format : XarrayIOFormat | None
            The xarray reading format, either 'zarr' or 'netcdf4'. Returns None if the extension is unsupported.
        """
        ext = self.get_reading_ext(uri)

        if ext in XarrayIOFormat.ZARR.extensions:
            return XarrayIOFormat.ZARR
        elif ext in XarrayIOFormat.NETCDF4.extensions:
            return XarrayIOFormat.NETCDF4
        return None

    def filter_uris_by_format(
        self, uris: list[str]
    ) -> tuple[list[str], XarrayIOFormat]:
        """Filter the list of URIs to only include those matching the specified format.

        Parameters
        ----------
        uris : list[str]
            List of URIs to filter.

        Returns
        -------
        filtered_uris : list[str]
            List of URIs that match the specified format.
        io_format : XarrayIOFormat
            The xarray reading format to filter by, either 'zarr' or 'netcdf4'.
        """
        io_format = self.get_io_format(uris[0])
        if io_format is None:
            raise ValueError(
                f"Unknown extension for XarrayDriverOptions: {self.get_reading_ext(uris[0])}. "
                "This can be overridden by setting the `ext_override` option to a supported extension (e.g. '.zarr' or '.nc')"
            )

        filtered_uris = []
        for _uri in uris:
            _format = self.get_io_format(_uri)
            if _format is None:
                logger.warning(
                    f"URI {_uri} has unsupported extension for format {io_format.value} and ext_override is not set, skipping..."
                )
            elif _format != io_format:
                logger.warning(
                    f"Reading {io_format.value} and {_uri} has a different format {_format.value}, skipping..."
                )
            else:
                filtered_uris.append(_uri)
        return filtered_uris, io_format

    def get_kwargs(self) -> dict[str, Any]:
        """Return attributes set that are not explicitly declared fields."""
        kwargs = super().get_kwargs()
        if kwargs.get("parallel") is True and kwargs.get("lock") is False:
            scheduler = get_scheduler(collections=[delayed(lambda: None)()])
            if getattr(scheduler, "__module__", None) == "dask.threaded":
                raise ValueError(
                    "Cannot use xarray open_mfdataset parallel=True with lock=False "
                    "with a threaded dask scheduler. Use lock=True or a non-threaded "
                    "scheduler such as distributed."
                )

        return kwargs


def _read_xarray(
    uris: list[str],
    options: XarrayDriverOptions,
    filesystem: FSSpecFileSystem,
    driver_name: str,
    handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
) -> xr.Dataset | None:
    """Read and merge xarray datasets from the given URIs based on the specified format.

    Parameters
    ----------
    uris : list[str]
        List of URIs to read data from.
    options : XarrayDriverOptions
        Driver options containing read configurations.
    filesystem : FSSpecFileSystem
        Filesystem object to access the URIs.
    driver_name : str
        Name of the driver for logging and error messages.
    handle_nodata : NoDataStrategy, optional
        Strategy to handle missing or empty data. Default is NoDataStrategy.RAISE.

    Returns
    -------
    xr.Dataset | None
        The merged xarray Dataset, or None if no data was found and the strategy allows.

    Raises
    ------
    ValueError
        If the file extension is unsupported.
    NoDataException
        If no data is found and the handling strategy requires raising an exception.
    """
    preprocessor = options.get_preprocessor()
    filtered_uris, io_format = options.filter_uris_by_format(uris)

    if io_format == XarrayIOFormat.ZARR:
        # FSMap contains the filesystem's credentials and storage_options.
        # xr.open_zarr raises a TypeError if 'storage_options' is passed alongside FSMap.
        read_kwargs = options.get_kwargs()
        extra_storage_options = read_kwargs.pop("storage_options", None)
        fsmaps = [
            filesystem.get_fsmap(uri, storage_options=extra_storage_options)
            for uri in filtered_uris
        ]
        datasets = [
            preprocessor(ds) for ds in open_zarrs(uris=fsmaps, read_kwargs=read_kwargs)
        ]
        ds: xr.Dataset = xr.merge(datasets)
    elif io_format == XarrayIOFormat.NETCDF4:
        ds: xr.Dataset = open_mfdataset(
            uris=filtered_uris,
            preprocessor=preprocessor,
            read_kwargs=options.get_kwargs(),
        )
    else:
        raise ValueError(
            f"Unknown extension for {driver_name}: {options.get_reading_ext(uris[0])} "
        )

    for variable in ds.data_vars:
        if ds[variable].size == 0:
            exec_nodata_strat(
                f"No data from driver: '{driver_name}' for variable: '{variable}'",
                strategy=handle_nodata,
            )
            return None  # handle_nodata == ignore

    return ds
