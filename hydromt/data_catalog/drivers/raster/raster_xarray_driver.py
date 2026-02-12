"""RasterDatasetDriver for zarr data."""

import logging
from pathlib import Path
from typing import Any, ClassVar

import xarray as xr
from aiohttp.client_exceptions import ClientResponseError
from pydantic import Field

from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers.base_driver import (
    DRIVER_OPTIONS_DESCRIPTION,
)
from hydromt.data_catalog.drivers.raster.raster_dataset_driver import (
    RasterDatasetDriver,
)
from hydromt.data_catalog.drivers.xarray_options import (
    XarrayDriverOptions,
    XarrayIOFormat,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.typing import (
    Geom,
    SourceMetadata,
    Variables,
    Zoom,
)

logger = logging.getLogger(__name__)


class RasterDatasetXarrayDriver(RasterDatasetDriver):
    """
    Driver for RasterDataset using the xarray library: ``raster_xarray``.

    Supports reading and writing zarr and netcdf files using xarray.
    zarr files will be read using `xr.open_zarr` and netcdf files using
    `xr.open_mfdataset`.

    """

    name = "raster_xarray"
    supports_writing = True
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = (
        XarrayIOFormat.ZARR.extensions | XarrayIOFormat.NETCDF4.extensions
    )
    options: XarrayDriverOptions = Field(
        default_factory=XarrayDriverOptions, description=DRIVER_OPTIONS_DESCRIPTION
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
        filtered_uris, io_format = self.options.filter_uris_by_format(uris)

        # Read and merge
        if io_format == XarrayIOFormat.ZARR:
            datasets = [
                preprocessor(ds)
                for ds in self._open_zarrs(filtered_uris, self.options.get_kwargs())
            ]
            ds: xr.Dataset = xr.merge(datasets)
        elif io_format == XarrayIOFormat.NETCDF4:
            ds: xr.Dataset = xr.open_mfdataset(
                filtered_uris,
                decode_coords="all",
                preprocess=preprocessor,
                **self.options.get_kwargs(),
                decode_timedelta=True,
            )
        else:
            raise ValueError(
                f"Unknown extension for RasterDatasetXarrayDriver: {self.options.get_reading_ext(uris[0])} "
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
        fmt = self.options.get_io_format(path)
        write_kwargs = write_kwargs or {}
        if fmt is None:
            logger.warning(
                f"Unknown extension for RasterDatasetXarrayDriver: {self.options.get_reading_ext(path)},"
                "switching to zarr"
            )
            fmt = XarrayIOFormat.ZARR
            path = Path(path).with_suffix(next(iter(XarrayIOFormat.ZARR.extensions)))
        if fmt == XarrayIOFormat.ZARR:
            write_kwargs.setdefault("zarr_format", 2)
            data.to_zarr(path, mode="w", **write_kwargs)
        else:
            data.to_netcdf(path, **write_kwargs)

        return Path(path)

    @staticmethod
    def _open_zarrs(uris: list[str], read_kwargs: dict[str, Any]) -> list[xr.Dataset]:
        """Open multiple zarr datasets with error handling."""
        datasets = []
        for _uri in uris:
            try:
                ds = xr.open_zarr(_uri, **read_kwargs)
                datasets.append(ds)
            except ClientResponseError as e:
                if e.status == 401:
                    raise PermissionError(
                        f"Unauthorized access to {_uri}. Check your credentials."
                    ) from e
                raise
        return datasets
