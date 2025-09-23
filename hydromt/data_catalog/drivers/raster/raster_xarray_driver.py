"""RasterDatasetDriver for zarr data."""

from functools import partial
from logging import Logger, getLogger
from os.path import splitext
from typing import Callable, ClassVar, List, Optional

import xarray as xr
from pydantic import Field

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
from hydromt.data_catalog.drivers.base_driver import DriverOptions
from hydromt.data_catalog.drivers.preprocessing import PREPROCESSORS
from hydromt.data_catalog.drivers.raster.raster_dataset_driver import (
    RasterDatasetDriver,
)

logger: Logger = getLogger(__name__)

_ZARR_EXT = ".zarr"
_NETCDF_EXT = [".nc", ".netcdf"]


class RasterXarrayOptions(DriverOptions):
    """Options for RasterXarrayDriver."""

    preprocess: Optional[str] = None
    """Name of preprocessor to apply before merging datasets. Available preprocessors include: round_latlon, to_datetimeindex, remove_duplicates, harmonise_dims. See their docstrings for details."""

    ext_override: Optional[str] = None
    """Override the file extension check and try to read all files as the given extension. Useful when reading zarr files without the .zarr extension."""

    def get_preprocessor(self) -> Optional[Callable]:
        """Get the preprocessor function."""
        if self.preprocess is None:
            return None
        preprocessor = PREPROCESSORS.get(self.preprocess)
        if not preprocessor:
            raise ValueError(f"unknown preprocessor: '{self.preprocess}'")
        return preprocessor

    def get_ext_override(self, uris: List[str]) -> Optional[str]:
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

    Driver **options** include:

    * preprocess: Optional[str], name of preprocessor to apply before merging datasets.
      Available preprocessors include: round_latlon, to_datetimeindex,
      remove_duplicates, harmonise_dims. See their docstrings for details.
    * ext_override: Optional[str], if set, will override the file extension check
      and try to read all files as the given extension. Useful when reading zarr
      files without the .zarr extension.
    * Any other option supported by `xr.open_zarr` or `xr.open_mfdataset`.

    """

    name = "raster_xarray"
    supports_writing = True
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {_ZARR_EXT, *_NETCDF_EXT}
    options: RasterXarrayOptions = Field(default_factory=RasterXarrayOptions)

    def read(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        zoom: Optional[Zoom] = None,
        chunks: Optional[dict] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
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
        preprocessor: Optional[Callable] = self.options.get_preprocessor()

        # Check for the override flag
        first_ext = self.options.get_ext_override(uris)

        # When is zarr, open like a zarr archive
        if first_ext == _ZARR_EXT:
            opn: Callable = partial(
                xr.open_zarr,
                **self.options.get_kwargs(),
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
                **self.options.get_kwargs(),
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
