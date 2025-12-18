"""Driver using rasterio for RasterDataset."""

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import rasterio
import rasterio.errors
import xarray as xr
from pydantic import Field, field_serializer, model_validator
from pyproj import CRS

from hydromt._utils import _strip_scheme, cache_vrt_tiles, temp_env
from hydromt.config import SETTINGS
from hydromt.data_catalog.drivers.base_driver import DriverOptions
from hydromt.data_catalog.drivers.raster import RasterDatasetDriver
from hydromt.error import NoDataException, NoDataStrategy, exec_nodata_strat
from hydromt.gis._gdal_drivers import GDAL_DRIVER_CODE_MAP
from hydromt.gis.gis_utils import zoom_to_overview_level
from hydromt.readers import open_mfraster
from hydromt.typing import (
    Geom,
    SourceMetadata,
    Variables,
    Zoom,
)

logger = logging.getLogger(__name__)

_TIFF_EXT = ".tif"


class RasterioOptions(DriverOptions):
    """Options for RasterioDriver."""

    KWARGS_FOR_OPEN: ClassVar[set[str]] = {"mosaic_kwargs"}

    mosaic: bool = Field(
        default=False,
        description="If True and multiple uris are given, will mosaic the datasets together using `rasterio.merge.merge`. Default is False.",
    )

    mosaic_kwargs: dict[str, Any] = Field(
        default={},
        description="Additional keyword arguments to pass to `rasterio.merge.merge`.",
    )

    cache: bool = Field(
        default=False,
        description="If True and reading from VRT files, will cache the tiles locally to speed up reading. Default is False.",
    )

    cache_root: str = Field(
        default=str(SETTINGS.cache_root),
        description="Root directory for caching. Default is taken from `hydromt.config.SETTINGS.cache_root`.",
    )

    cache_dir: str | None = Field(
        default=None,
        description="Subdirectory for caching. Default is the stem of the first uri without extension.",
    )

    def get_cache_path(self, uris: list[str]) -> Path:
        """Get the cache path based on the options and uris."""
        if self.cache_dir is not None:
            cache_dir = Path(self.cache_root) / self.cache_dir
        else:
            # default to first uri without extension
            cache_dir = Path(self.cache_root) / Path(_strip_scheme(uris[0])[1]).stem
        return cache_dir

    @model_validator(mode="after")
    def _convert_path_to_str(self):
        """Convert Path to str for pydantic compatibility."""
        if isinstance(self.cache_root, Path):
            self.cache_root = self.cache_root.as_posix()
        if isinstance(self.cache_dir, Path):
            self.cache_dir = self.cache_dir.as_posix()
        return self

    @field_serializer("cache_root", "cache_dir")
    def serialize_paths(self, value: Path) -> str | None:
        """Serialize Path to str for pydantic compatibility."""
        if value is None:
            return None
        return Path(value).as_posix()


class RasterioDriver(RasterDatasetDriver):
    """
    Driver for RasterDataset using the rasterio library: ``rasterio``.

    Supports reading and writing raster files using rasterio.
    """

    name = "rasterio"
    supports_writing = True
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {
        "." + extension for extension in GDAL_DRIVER_CODE_MAP.keys()
    }
    options: RasterioOptions = Field(default_factory=RasterioOptions)

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
        Read raster data using the rasterio library.

        Supports reading single or multiple raster files (optionally mosaicked),
        applying spatial masks, caching VRT tiles, and reading overviews at
        different zoom levels. Returns an xarray Dataset constructed from raster bands.

        Parameters
        ----------
        uris : list[str]
            List of raster file URIs to read.
        handle_nodata : NoDataStrategy, optional
            Strategy for handling missing or empty data. Default is NoDataStrategy.RAISE.
        mask : Geom | None, optional
            Geometry used to mask or clip the raster data. Default is None.
        variables : Variables | None, optional
            List of variables or band names to read. Default is None.
        zoom : Zoom | None, optional
            Requested zoom level or resolution. Used to determine the appropriate overview level. Default is None.
        chunks : dict[str, Any] | None, optional
            Dask chunking configuration for lazy loading. Default is None.
        metadata : SourceMetadata | None, optional
            Optional metadata describing CRS, nodata, and overview levels. Default is None.

        Returns
        -------
        xr.Dataset
            The loaded raster dataset as an xarray Dataset.

        Raises
        ------
        ValueError
            If the file extension is unsupported or invalid.
        rasterio.errors.RasterioIOError
            If an I/O error occurs during reading.
        """
        if len(uris) == 0:
            return None  # handle_nodata == ignore

        if metadata is None:
            metadata = SourceMetadata()

        # Caching portion, only when the flag is True and the file format is vrt
        if all(uri.endswith(".vrt") for uri in uris) and self.options.cache:
            cache_dir: Path = self.options.get_cache_path(uris)
            uris_cached = []
            for uri in uris:
                cached_uri = cache_vrt_tiles(
                    uri, geom=mask, fs=self.filesystem.get_fs(), cache_dir=cache_dir
                )
                uris_cached.append(cached_uri)
            uris = uris_cached

        if mask is not None:
            self.options.mosaic_kwargs.update({"mask": mask})

        open_kwargs = self.options.get_kwargs()
        if np.issubdtype(type(metadata.nodata), np.number):
            open_kwargs.update({"nodata": metadata.nodata})

        # Fix overview level
        if zoom:
            try:
                zls_dict: dict[int, float] = metadata.zls_dict
                crs: CRS | None = metadata.crs
            except AttributeError:  # pydantic extra=allow on SourceMetadata
                zls_dict, crs = self._get_zoom_levels_and_crs(uris[0])

            overview_level: int | None = zoom_to_overview_level(
                zoom, mask, zls_dict, crs
            )
            if overview_level:
                # NOTE: overview levels start at zoom_level 1, see _get_zoom_levels_and_crs
                open_kwargs.update(overview_level=overview_level - 1)

        if chunks is not None:
            open_kwargs.update({"chunks": chunks})

        mosaic: bool = self.options.mosaic and len(uris) > 1
        mosaic_kwargs = open_kwargs.pop("mosaic_kwargs", {})
        if mosaic_kwargs and not mosaic:
            logger.warning(
                "mosaic_kwargs provided but mosaic is False. Ignoring mosaic_kwargs. To use mosaic_kwargs, set mosaic=True in driver options."
            )

        # If the metadata resolver has already resolved the overview level,
        # trying to open zoom levels here will result in an error.
        # Better would be to separate uriresolver and driver: https://github.com/Deltares/hydromt/issues/1023
        # Then we can implement looking for a overview level in the driver.
        def _open() -> xr.Dataset:
            try:
                return open_mfraster(
                    uris, mosaic=mosaic, mosaic_kwargs=mosaic_kwargs, **open_kwargs
                )
            except rasterio.errors.RasterioIOError as e:
                if "Cannot open overview level" in str(e):
                    open_kwargs.pop("overview_level", None)
                    return open_mfraster(
                        uris, mosaic=mosaic, mosaic_kwargs=mosaic_kwargs, **open_kwargs
                    )
                else:
                    raise

        # rasterio uses specific environment variable for s3 access.
        try:
            anon: str = self.filesystem.get_fs().anon
        except AttributeError:
            anon = ""

        if anon:
            with temp_env(**{"AWS_NO_SIGN_REQUEST": "true"}):
                ds = _open()
        else:
            ds = _open()

        # Mosaic's can mess up the chunking, which can error during writing
        # Or maybe setting
        chunks = open_kwargs.get("chunks", None)
        if chunks is not None:
            ds = ds.chunk(chunks=chunks)

        # rename ds with single band if single variable is requested
        if variables is not None and len(variables) == 1 and len(ds.data_vars) == 1:
            ds = ds.rename({list(ds.data_vars.keys())[0]: list(variables)[0]})

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
        data: xr.Dataset | xr.DataArray,
        *,
        write_kwargs: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write a RasterDataset to disk using the rasterio library.

        This method is not implemented in this driver. Concrete implementations
        must provide a way to write raster datasets to supported formats.

        Parameters
        ----------
        path : Path | str
            Destination path for the raster dataset.
        data : xr.DataArray | xr.Dataset
            The xarray DataArray or Dataset to write.
        write_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments for writing. Default is None.

        Returns
        -------
        Path
            The path to the written raster dataset.
        """
        path = Path(path)
        write_kwargs = write_kwargs or {}
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unknown extension for RasterioDriver: {path.suffix}")

        if path.suffix == ".vrt":
            logger.warning(
                "Writing to VRT format is not supported by RasterioDriver, will attempt to write as GeoTIFF instead."
            )
            path = path.with_suffix(_TIFF_EXT)

        gdal_driver = GDAL_DRIVER_CODE_MAP.get(path.suffix.lstrip(".").lower())

        if "*" in str(path) and isinstance(data, xr.DataArray):
            if len(data.dims) < 3:
                raise ValueError(
                    "Writing multiple files with wildcard requires at least 3 dimensions in data array"
                )

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.name.count("*") != 1:
                raise ValueError(
                    "There must be exactly one wildcard `*` in the filename when multiple outputs required"
                )

            dim0 = data.dims[0]

            for label in data[dim0]:
                ds_sel = data.sel({dim0: label})
                file_name = path.name.replace("*", f"{label.values}")

                self._write_raster(
                    ds_sel, gdal_driver, path.with_name(file_name), **write_kwargs
                )

            return path
        if isinstance(data, xr.Dataset):
            if len(data.data_vars) == 1:
                data = data[list(data.data_vars.keys())[0]]
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                for var in data.data_vars:
                    if "*" in path.name:
                        file_name = path.name.replace("*", var)
                        file_path = path.with_name(file_name)
                    else:
                        file_path = path.parent / f"{var}{path.suffix}"
                    data_raster = data[var]
                    self._write_raster(
                        data_raster, gdal_driver, file_path, **write_kwargs
                    )
                return path if "*" in path.name else path.parent / f"*{path.suffix}"
        self._write_raster(data, gdal_driver, path, **write_kwargs)
        return path

    @staticmethod
    def _get_zoom_levels_and_crs(uri: str) -> tuple[dict[int, float], int]:
        """Get zoom levels and crs from adapter or detect from tif file if missing."""
        zoom_levels = {}
        crs = None
        try:
            with rasterio.open(uri) as src:
                res = abs(src.res[0])
                crs = src.crs
                overviews = [src.overviews(i) for i in src.indexes]
                if len(overviews[0]) > 0:  # check overviews for band 0
                    # check if identical
                    if not all([o == overviews[0] for o in overviews]):
                        raise ValueError("Overviews are not identical across bands")
                    # dict with overview level and corresponding resolution
                    zls = [1] + overviews[0]
                    zoom_levels = {i: res * zl for i, zl in enumerate(zls)}
        except rasterio.RasterioIOError as e:
            logger.warning(f"IO error while detecting zoom levels: {e}")
        return zoom_levels, crs

    def _write_raster(
        self, data: xr.DataArray, driver: str, path: Path, **write_kwargs: Any
    ) -> None:
        """Write raster data to file using rasterio."""
        y_coords = data[data.raster.y_dim]
        x_coords = data[data.raster.x_dim]
        if (
            y_coords.size < 2
            or (y_coords.ndim == 2 and y_coords.shape[0] < 2)
            or x_coords.size < 2
            or (x_coords.ndim == 2 and x_coords.shape[1] < 2)
        ):
            raise NoDataException(
                f"Cannot write raster data with insufficient spatial dimensions: {data.raster.y_dim} size {y_coords.size}, {data.raster.x_dim} size {x_coords.size}",
            )
        data.raster.to_raster(path, driver=driver, **write_kwargs)
