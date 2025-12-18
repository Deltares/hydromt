"""GeoDatasetDriver for zarr data."""

import logging
from functools import partial
from os.path import splitext
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
    GeoDatasetOptions,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.typing import (
    Geom,
    Predicate,
    SourceMetadata,
)

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
        mask: Geom | None = None,
        predicate: Predicate = "intersects",
        metadata: SourceMetadata | None = None,
    ) -> xr.Dataset:
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
        preprocessor = self.options.get_preprocessor()
        first_ext = splitext(uris[0])[-1]

        if first_ext == _ZARR_EXT:
            opn = partial(xr.open_zarr, **self.options.get_kwargs())
            datasets = []
            for _uri in uris:
                ext = splitext(_uri)[-1]
                if ext != first_ext:
                    logger.warning(f"Reading zarr and {_uri} was not, skipping...")
                else:
                    datasets.append(preprocessor(opn(_uri)))

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
                filtered_uris,
                decode_coords="all",
                preprocess=preprocessor,
                **self.options.get_kwargs(),
                decode_timedelta=True,
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
        path = Path(path)
        ext = path.suffix
        write_kwargs = write_kwargs or {}
        if ext == _ZARR_EXT:
            write_kwargs.setdefault("zarr_format", 2)
            data.vector.to_zarr(path, **write_kwargs)
        elif ext in _NETCDF_EXT:
            data.vector.to_netcdf(path, **write_kwargs)
        else:
            raise ValueError(f"Unknown extension for GeoDatasetXarrayDriver: {ext} ")

        return path
