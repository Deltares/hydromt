"""DatasetDriver for zarr data."""

import logging
from os.path import splitext
from pathlib import Path
from typing import Any, ClassVar

import xarray as xr
from pydantic import Field

from hydromt.data_catalog.drivers.base_driver import (
    DRIVER_OPTIONS_DESCRIPTION,
    DriverOptions,
)
from hydromt.data_catalog.drivers.dataset.dataset_driver import DatasetDriver
from hydromt.data_catalog.drivers.preprocessing import Preprocessor, get_preprocessor
from hydromt.error import NoDataStrategy, exec_nodata_strat

logger = logging.getLogger(__name__)


_ZARR_EXT = ".zarr"
_NETCDF_EXT = [".nc", ".netcdf"]


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

    def get_preprocessor(self) -> Preprocessor:
        """Get the preprocessor function."""
        return get_preprocessor(self.preprocess)

    def get_reading_ext(self, uri: str) -> str:
        """Get the file extension to use for reading, can be overridden."""
        if not self.ext_override:
            return splitext(uri)[-1]
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
        first_ext = self.options.get_reading_ext(uris[0])

        # Determine reading extensions based on first file
        if first_ext in _NETCDF_EXT:
            reading_extentions = set(_NETCDF_EXT)
        elif first_ext == _ZARR_EXT:
            reading_extentions = {_ZARR_EXT}
        else:
            raise ValueError(f"Unknown extension for DatasetXarrayDriver: {first_ext}")

        # Filter uris based on extension
        filtered_uris = []
        for _uri in uris:
            ext = self.options.get_reading_ext(_uri)
            if ext not in reading_extentions:
                logger.warning(
                    f"Reading {reading_extentions} and {_uri} has a different extension, skipping..."
                )
            else:
                filtered_uris.append(_uri)

        # Read and merge
        if first_ext == ".zarr":
            datasets = [
                preprocessor(xr.open_zarr(_uri, **self.options.get_kwargs()))
                for _uri in filtered_uris
            ]
            ds: xr.Dataset = xr.merge(datasets)
        elif first_ext in [".nc", ".netcdf"]:
            ds: xr.Dataset = xr.open_mfdataset(
                filtered_uris,
                decode_coords="all",
                preprocess=preprocessor,
                **self.options.get_kwargs(),
                decode_timedelta=True,
            )
        else:
            raise ValueError(f"Unknown extension for DatasetXarrayDriver: {first_ext}")

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
        path = Path(path)
        ext = path.suffix
        write_kwargs = write_kwargs or {}
        if ext == ".zarr":
            write_kwargs.setdefault("zarr_format", 2)
            data.to_zarr(path, **write_kwargs)
        elif ext in [".nc", ".netcdf"]:
            data.to_netcdf(path, **write_kwargs)
        else:
            raise ValueError(f"Unknown extension for DatasetXarrayDriver: {ext} ")

        return path
