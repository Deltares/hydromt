"""RasterDatasetDriver for zarr data."""

import logging
from functools import partial
from os.path import splitext
from pathlib import Path
from typing import Any, Callable, ClassVar

import xarray as xr
from pydantic import Field

from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers.base_driver import (
    DRIVER_OPTIONS_DESCRIPTION,
    DriverOptions,
)
from hydromt.data_catalog.drivers.preprocessing import Preprocessor, get_preprocessor
from hydromt.data_catalog.drivers.raster.raster_dataset_driver import (
    RasterDatasetDriver,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.typing import (
    Geom,
    SourceMetadata,
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

    def get_preprocessor(self) -> Preprocessor:
        """Get the preprocessor function."""
        return get_preprocessor(self.preprocess)

    def get_reading_ext(self, uri: str) -> str:
        """Get the file extension to use for reading, can be overridden."""
        if not self.ext_override:
            return splitext(uri)[-1]
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
        mask: Geom | None = None,
        variables: Variables | None = None,
        zoom: Zoom | None = None,
        chunks: dict[str, Any] | None = None,
        metadata: SourceMetadata | None = None,
    ) -> xr.Dataset:
        """
        Read zarr or netCDF raster data into an xarray Dataset.

        Supports both zarr archives and NetCDF datasets via `xr.open_zarr` and
        `xr.open_mfdataset`. Optionally applies a preprocessing function defined
        in the driver options. Unused parameters (e.g., mask, zoom) are ignored
        but logged for transparency.

        Parameters
        ----------
        uris : list[str]
            List of URIs pointing to zarr or netCDF files.
        handle_nodata : NoDataStrategy, optional
            Strategy for handling missing or empty data. Default is NoDataStrategy.RAISE.
        mask : Geom | None, optional
            Spatial mask or geometry (currently unused). Default is None.
        variables : Variables | None, optional
            List of variables to select from the dataset (currently unused). Default is None.
        zoom : Zoom | None, optional
            Zoom level or resolution (currently unused). Default is None.
        chunks : dict[str, Any] | None, optional
            Chunking configuration for Dask-based reading (currently unused). Default is None.
        metadata : SourceMetadata | None, optional
            Optional metadata about the dataset source (currently unused). Default is None.

        Returns
        -------
        xr.Dataset
            The merged xarray Dataset.

        Raises
        ------
        ValueError
            If the file extension is unsupported.

        Warning
        -------
        The `mask`, `variables`, `zoom`, `chunks` and `metadata` parameters are not used directly in this driver,
        but are included for consistency with the GeoDataFrameDriver interface.
        """
        _warn_on_unused_kwargs(
            self.__class__.__name__,
            {
                "mask": mask,
                "variables": variables,
                "zoom": zoom,
                "chunks": chunks,
                "metadata": metadata,
            },
        )

        if len(uris) == 0:
            return None  # handle_nodata == ignore

        preprocessor = self.options.get_preprocessor()
        first_ext = self.options.get_reading_ext(uris[0])

        # Filter uris based on extension
        filtered_uris = []
        for _uri in uris:
            ext = self.options.get_reading_ext(_uri)
            if ext != first_ext:
                logger.warning(
                    f"Reading {first_ext} and {_uri} has a different extension, skipping..."
                )
            else:
                filtered_uris.append(_uri)

        # Read and merge
        if first_ext == _ZARR_EXT:
            opn: Callable = partial(xr.open_zarr, **self.options.get_kwargs())
            datasets = [preprocessor(opn(_uri)) for _uri in filtered_uris]
            ds: xr.Dataset = xr.merge(datasets)
        elif first_ext in _NETCDF_EXT:
            ds: xr.Dataset = xr.open_mfdataset(
                filtered_uris,
                decode_coords="all",
                preprocess=preprocessor,
                **self.options.get_kwargs(),
                decode_timedelta=True,
            )
        else:
            raise ValueError(
                f"Unknown extension for RasterDatasetXarrayDriver: {first_ext}"
            )

        for variable in ds.data_vars:
            if ds[variable].size == 0:
                exec_nodata_strat(
                    f"No data from driver: '{self.name}' for variable: '{variable}'",
                    strategy=handle_nodata,
                )
                return None  # handle_nodata == ignore
        return ds

    def write(
        self,
        path: Path | str,
        data: xr.Dataset,
        *,
        write_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write a RasterDataset to disk using Zarr or NetCDF format.

        Supports writing datasets to `.zarr`, `.nc`, or `.netcdf` formats depending
        on the file extension. If an unsupported extension is provided, defaults to Zarr.

        Parameters
        ----------
        path : Path | str
            Destination path for the dataset.
        data : xr.Dataset
            The xarray Dataset to write.
        write_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments passed to `to_zarr` or `to_netcdf`. Default is None.

        Returns
        -------
        Path
            The path to the written dataset.

        Raises
        ------
        ValueError
            If the file extension is not recognized or supported.
        """
        no_ext, ext = splitext(path)
        write_kwargs = write_kwargs or {}
        # set filepath if incompat
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning(
                f"Unknown extension for RasterDatasetXarrayDriver: {ext},"
                "switching to zarr"
            )
            path = no_ext + _ZARR_EXT
            ext = _ZARR_EXT
        if ext == _ZARR_EXT:
            write_kwargs.setdefault("zarr_format", 2)
            data.to_zarr(path, mode="w", **write_kwargs)
        else:
            data.to_netcdf(path, **write_kwargs)

        return Path(path)
