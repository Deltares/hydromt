"""DatasetDriver for zarr data."""

import logging
from pathlib import Path
from typing import Any, ClassVar

import xarray as xr
from pydantic import Field

from hydromt.data_catalog.drivers.base_driver import (
    DRIVER_OPTIONS_DESCRIPTION,
)
from hydromt.data_catalog.drivers.dataset.dataset_driver import DatasetDriver
from hydromt.data_catalog.drivers.xarray_options import (
    XarrayDriverOptions,
    XarrayIOFormat,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.readers import open_mfdataset, open_zarrs

logger = logging.getLogger(__name__)


class DatasetXarrayDriver(DatasetDriver):
    """
    Driver for Dataset using xarray: ``dataset_xarray``.

    Supports reading and writing zarr and netcdf files using xarray.
    zarr files will be read using `xr.open_zarr` and netcdf files using
    `xr.open_mfdataset`.

    """

    name: ClassVar[str] = "dataset_xarray"
    supports_writing: ClassVar[bool] = True
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = (
        XarrayIOFormat.ZARR.extensions | XarrayIOFormat.NETCDF4.extensions
    )

    options: XarrayDriverOptions = Field(
        default_factory=XarrayDriverOptions, description=DRIVER_OPTIONS_DESCRIPTION
    )

    def read(
        self, uris: list[str], *, handle_nodata: NoDataStrategy = NoDataStrategy.RAISE
    ) -> xr.Dataset:
        """
        Read zarr or netCDF data into an xarray Dataset.

        Supports reading multiple compatible datasets and merging them into a single
        xarray Dataset. File format is automatically inferred from the file extension,
        unless overridden via the driver options. Optionally applies a preprocessor
        function to each dataset before merging.

        Parameters
        ----------
        uris : list[str]
            List of URIs to read data from. All files must share the same format.
        handle_nodata : NoDataStrategy, optional
            Strategy to handle missing or empty data. Default is NoDataStrategy.RAISE.

        Returns
        -------
        xr.Dataset
            The dataset read from the source files.

        Raises
        ------
        ValueError
            If the provided files have mixed or unsupported extensions.
        """
        preprocessor = self.options.get_preprocessor()
        filtered_uris, io_format = self.options.filter_uris_by_format(uris)

        # Read and merge
        if io_format == XarrayIOFormat.ZARR:
            # FSMap contains the filesystem's credentials and storage_options.
            # xr.open_zarr raises a TypeError if 'storage_options' is passed alongside FSMap.
            read_kwargs = self.options.get_kwargs()
            extra_storage_options = read_kwargs.pop("storage_options", None)
            fsmaps = [
                self.filesystem.get_fsmap(uri, storage_options=extra_storage_options)
                for uri in filtered_uris
            ]
            datasets = [
                preprocessor(ds)
                for ds in open_zarrs(uris=fsmaps, read_kwargs=read_kwargs)
            ]
            ds: xr.Dataset = xr.merge(datasets)
        elif io_format == XarrayIOFormat.NETCDF4:
            ds: xr.Dataset = open_mfdataset(
                uris=filtered_uris,
                preprocessor=preprocessor,
                read_kwargs=self.options.get_kwargs(),
            )
        else:
            raise ValueError(
                f"Unknown extension for DatasetXarrayDriver: {self.options.get_reading_ext(uris[0])}"
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
        Write an xarray Dataset to disk using the xarray I/O engine.

        Supports writing to both Zarr and NetCDF formats. The file format is inferred
        from the file extension. If the extension is not recognized, a ValueError is raised.

        Parameters
        ----------
        path : Path | str
            Destination path or URI where the Dataset will be written.
            The file extension determines the output format:
            `.zarr`, `.nc`, or `.netcdf`.
        data : xr.Dataset
            The Dataset to write to disk.
        write_kwargs : dict[str, Any], optional
            Additional keyword arguments passed to the xarray write function
            (`Dataset.to_zarr` or `Dataset.to_netcdf`). Default is None.

        Returns
        -------
        Path
            The path where the dataset was written.

        Raises
        ------
        ValueError
            If the provided file extension is unsupported.
        """
        fmt = self.options.get_io_format(path)
        write_kwargs = write_kwargs or {}
        if fmt == XarrayIOFormat.ZARR:
            write_kwargs.setdefault("zarr_format", 2)
            data.to_zarr(path, **write_kwargs)
        elif fmt == XarrayIOFormat.NETCDF4:
            data.to_netcdf(path, **write_kwargs)
        else:
            raise ValueError(
                f"Unknown extension for DatasetXarrayDriver: {self.options.get_reading_ext(path)}"
            )

        return path
