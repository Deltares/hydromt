"""GeoDatasetDriver for zarr data."""

import logging
from pathlib import Path
from typing import Any, ClassVar

import xarray as xr
from pydantic import Field

from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.data_catalog.drivers.base_driver import (
    DRIVER_OPTIONS_DESCRIPTION,
)
from hydromt.data_catalog.drivers.geodataset.geodataset_driver import (
    GeoDatasetDriver,
)
from hydromt.data_catalog.drivers.xarray_options import (
    XarrayDriverOptions,
    XarrayIOFormat,
    _read_xarray,
)
from hydromt.error import NoDataStrategy
from hydromt.typing import (
    Geom,
    Predicate,
    SourceMetadata,
)

logger = logging.getLogger(__name__)


class GeoDatasetXarrayDriver(GeoDatasetDriver):
    """
    Driver for GeoDataset using the xarray library: ``geodataset_xarray``.

    Supports reading and writing zarr and netcdf files using xarray.
    zarr files will be read using `xr.open_zarr` and netcdf files using
    `xr.open_mfdataset`.
    """

    name: ClassVar[str] = "geodataset_xarray"
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
        predicate: Predicate = "intersects",
        metadata: SourceMetadata | None = None,
    ) -> xr.Dataset | None:
        """
        Read in data to an xarray Dataset.

        Parameters
        ----------
        uris : list[str]
            List of URIs to read data from.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle missing data. Default is NoDataStrategy.RAISE.
        mask : Geom | None, optional
            Optional spatial mask to clip the dataset.
        predicate : Predicate, optional
            Spatial predicate for filtering geometries. Default is "intersects".
        metadata : SourceMetadata | None, optional
            Optional metadata object to attach to the loaded dataset.

        Returns
        -------
        xr.Dataset | None
            The dataset read from the source, or None if no data found and strategy allows.

        Warning
        -------
        The `mask`, `predicate` and `metadata` parameters are not used directly in this driver,
        but are included for consistency with the GeoDataFrameDriver interface.
        """
        _warn_on_unused_kwargs(
            self.__class__.__name__,
            {
                "mask": mask,
                "predicate": predicate,
                "metadata": metadata,
            },
        )
        return _read_xarray(
            uris=uris,
            options=self.options,
            filesystem=self.filesystem,
            driver_name=self.name,
            handle_nodata=handle_nodata,
        )

    def write(
        self,
        path: Path | str,
        data: xr.Dataset,
        *,
        write_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write a GeoDataset to disk in Zarr or NetCDF format.

        Parameters
        ----------
        path : Path | str
            Destination path or URI where the dataset will be written. Must end with a
            supported extension ('.zarr', '.nc', or '.netcdf').
        data : xr.Dataset
            The xarray Dataset to write.
        write_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments passed to the underlying write function.
            For example, `encoding` for NetCDF, or `mode` for Zarr. Default is None.

        Returns
        -------
        Path
            The path where the dataset was written.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        fmt = self.options.get_io_format(path)
        write_kwargs = write_kwargs or {}
        if fmt == XarrayIOFormat.ZARR:
            write_kwargs.setdefault("zarr_format", 2)
            data.vector.to_zarr(path, **write_kwargs)
        elif fmt == XarrayIOFormat.NETCDF4:
            data.vector.to_netcdf(path, **write_kwargs)
        else:
            raise ValueError(
                f"Unknown extension for GeoDatasetXarrayDriver: {self.options.get_reading_ext(path)}"
            )

        return path
