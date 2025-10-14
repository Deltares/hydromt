"""RasterDatasetDriver for zarr data."""

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
from hydromt.data_catalog.drivers.preprocessing import get_preprocessor
from hydromt.data_catalog.drivers.raster.raster_dataset_driver import (
    RasterDatasetDriver,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.typing import (
    Geom,
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
    Zoom,
)

logger = logging.getLogger(__name__)

_ZARR_EXT = ".zarr"
_NETCDF_EXT = [".nc", ".netcdf"]


class RasterXarrayOptions(DriverOptions):
    """Options for RasterXarrayDriver."""

    preprocess: str | None = None
    """Name of preprocessor to apply before merging datasets. Available preprocessors include: round_latlon, to_datetimeindex, remove_duplicates, harmonise_dims. See their docstrings for details."""

    ext_override: str | None = None
    """Override the file extension check and try to read all files as the given extension. Useful when reading zarr files without the .zarr extension."""

    def get_preprocessor(self) -> Callable | None:
        """Get the preprocessor function."""
        if self.preprocess is None:
            return None
        return get_preprocessor(self.preprocess)

    def get_ext_override(self, uris: list[str]) -> str | None:
        """Get the extension override."""
        if not self.ext_override:
            return splitext(uris[0])[-1]
        return self.ext_override


class RasterDatasetXarrayDriver(RasterDatasetDriver):
    """
    Driver for RasterDataset using the xarray library: ``raster_xarray``.

    Supports reading and writing zarr and netcdf files using xarray.
    zarr files will be read using `xr.open_zarr` and netcdf files using
    `xr.open_mfdataset`.

    """

    name = "raster_xarray"
    supports_writing = True
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {_ZARR_EXT, *_NETCDF_EXT}
    options: RasterXarrayOptions = Field(
        default_factory=RasterXarrayOptions, description=DRIVER_OPTIONS_DESCRIPTION
    )

    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        kwargs_for_open: dict[str, Any] | None = None,
        mask: Geom | None = None,
        variables: Variables | None = None,
        time_range: TimeRange | None = None,
        zoom: Zoom | None = None,
        chunks: dict[str, Any] | None = None,
        metadata: SourceMetadata | None = None,
    ) -> xr.Dataset:
        """
        Read zarr data to an xarray DataSet.

        Args:
        """
        _warn_on_unused_kwargs(
            self.__class__.__name__,
            {
                "zoom": zoom,
            },
        )

        # Sort out the preprocessor
        preprocessor: Callable | None = self.options.get_preprocessor()

        # Check for the override flag
        first_ext = self.options.get_ext_override(uris)

        # When is zarr, open like a zarr archive
        kwargs_for_open = kwargs_for_open or {}
        if first_ext == _ZARR_EXT:
            opn: Callable = partial(
                xr.open_zarr,
                **self.options.get_kwargs() | kwargs_for_open,
            )
            datasets = []
            for _uri in uris:
                ext = splitext(_uri)[-1]
                if ext != first_ext and not self.options.ext_override:
                    logger.warning(f"Reading zarr and {_uri} was not, skipping...")
                else:
                    datasets.append(
                        preprocessor(opn(_uri)) if preprocessor else opn(_uri)
                    )

            ds: xr.Dataset = xr.merge(datasets)

        # Normal netcdf file(s)
        elif first_ext in _NETCDF_EXT:
            filtered_uris = []
            for _uri in uris:
                ext = splitext(_uri)[-1]
                if ext != first_ext:
                    logger.warning(f"Reading netcdf and {_uri} was not, skipping...")
                else:
                    filtered_uris.append(_uri)
            ds: xr.Dataset = xr.open_mfdataset(
                filtered_uris,
                decode_coords="all",
                preprocess=preprocessor,
                **self.options.get_kwargs() | kwargs_for_open,
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
        if ext not in self.SUPPORTED_EXTENSIONS:
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
