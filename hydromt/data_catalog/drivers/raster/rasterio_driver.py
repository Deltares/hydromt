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

from hydromt._utils import _cache_vrt_tiles, _strip_scheme, temp_env
from hydromt.config import SETTINGS
from hydromt.data_catalog.drivers.base_driver import DriverOptions
from hydromt.data_catalog.drivers.raster import RasterDatasetDriver
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.gis.gis_utils import zoom_to_overview_level
from hydromt.readers import open_mfraster
from hydromt.typing import (
    Geom,
    SourceMetadata,
    Variables,
    Zoom,
)

logger = logging.getLogger(__name__)


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
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {
        "." + extension
        for extension in rasterio.drivers.raster_driver_extensions()
        if extension != "nc"
    }  # Exclude netcdf as a supported file type
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
        if metadata is None:
            metadata = SourceMetadata()

        # Caching portion, only when the flag is True and the file format is vrt
        if all([uri.endswith(".vrt") for uri in uris]) and self.options.cache:
            cache_dir: Path = self.options.get_cache_path(uris)
            uris_cached = []
            for uri in uris:
                cached_uri = _cache_vrt_tiles(
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

        # If the metadata resolver has already resolved the overview level,
        # trying to open zoom levels here will result in an error.
        # Better would be to separate uriresolver and driver: https://github.com/Deltares/hydromt/issues/1023
        # Then we can implement looking for a overview level in the driver.
        def _open() -> xr.Dataset:
            try:
                return open_mfraster(uris, mosaic=mosaic, **open_kwargs)
            except rasterio.errors.RasterioIOError as e:
                if "Cannot open overview level" in str(e):
                    open_kwargs.pop("overview_level", None)
                    return open_mfraster(uris, mosaic=mosaic, **open_kwargs)
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
        return ds

    def write(
        self,
        path: Path | str,
        data: xr.Dataset,
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
        data : xr.Dataset
            The xarray Dataset to write.
        write_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments for writing. Default is None.

        Raises
        ------
        NotImplementedError
            Always raised because writing is not supported in this driver.
        """
        raise NotImplementedError()

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
