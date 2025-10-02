"""Driver using rasterio for RasterDataset."""

import copy
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import rasterio.errors
import xarray as xr
from pyproj import CRS

from hydromt._typing import (
    Geom,
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
    Zoom,
)
from hydromt._utils.caching import _cache_vrt_tiles
from hydromt._utils.temp_env import temp_env
from hydromt._utils.uris import _strip_scheme
from hydromt.config import SETTINGS
from hydromt.data_catalog.drivers.raster.raster_dataset_driver import (
    RasterDatasetDriver,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.gis.gis_utils import zoom_to_overview_level
from hydromt.io.readers import open_mfraster

logger: Logger = getLogger(__name__)


class RasterioDriver(RasterDatasetDriver):
    """
    Driver for RasterDataset using the rasterio library: ``rasterio``.

    Supports reading and writing raster files using rasterio.

    Driver **options** include:

    * mosaic: bool, if True and multiple uris are given, will mosaic the datasets
      together using `rasterio.merge.merge`. Default is False.
    * mosaic_kwargs: dict, additional keyword arguments to pass to
      `rasterio.merge.merge`.
    * cache: bool, if True and reading from VRT files, will cache the tiles
      locally to speed up reading. Default is False.
    * cache_root: str, root directory for caching. Default is taken from
      `hydromt.config.SETTINGS.cache_root`.
    * cache_dir: str, subdirectory for caching. Default is the stem of the first
      uri without extension.
    * Any other option supported by `hydromt.io.readers.open_mfraster`.

    """

    name = "rasterio"
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {
        "." + extension
        for extension in rasterio.drivers.raster_driver_extensions()
        if extension != "nc"
    }  # Exclude netcdf as a supported file type

    def read(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        time_range: Optional[TimeRange] = None,
        variables: Optional[Variables] = None,
        zoom: Optional[Zoom] = None,
        chunks: Optional[dict] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> xr.Dataset:
        """Read data using rasterio."""
        if metadata is None:
            metadata = SourceMetadata()
        # build up kwargs for open_raster
        options = copy.deepcopy(self.options)
        mosaic_kwargs: Dict[str, Any] = self.options.get("mosaic_kwargs", {})
        mosaic: bool = options.pop("mosaic", False) and len(uris) > 1

        # get source-specific options
        cache_root: str = str(
            options.pop("cache_root", SETTINGS.cache_root),
        )

        # Check for caching, default to false
        cache_flag = options.pop("cache", False)

        # Caching portion, only when the flag is True and the file format is vrt
        if all([uri.endswith(".vrt") for uri in uris]) and cache_flag:
            cache_dir = Path(cache_root) / options.pop(
                "cache_dir",
                Path(
                    _strip_scheme(uris[0])[1]
                ).stem,  # default to first uri without extension
            )
            uris_cached = []
            for uri in uris:
                cached_uri: str = _cache_vrt_tiles(
                    uri, geom=mask, fs=self.filesystem, cache_dir=cache_dir
                )
                uris_cached.append(cached_uri)
            uris = uris_cached

        if mask is not None:
            mosaic_kwargs.update({"mask": mask})

        # get mosaic kwargs
        if mosaic_kwargs:
            options.update({"mosaic_kwargs": mosaic_kwargs})

        if np.issubdtype(type(metadata.nodata), np.number):
            options.update(nodata=metadata.nodata)

        # Fix overview level
        if zoom:
            try:
                zls_dict: Dict[int, float] = metadata.zls_dict
                crs: Optional[CRS] = metadata.crs
            except AttributeError:  # pydantic extra=allow on SourceMetadata
                zls_dict, crs = self._get_zoom_levels_and_crs(uris[0])

            overview_level: Optional[int] = zoom_to_overview_level(
                zoom, mask, zls_dict, crs
            )
            if overview_level:
                # NOTE: overview levels start at zoom_level 1, see _get_zoom_levels_and_crs
                options.update(overview_level=overview_level - 1)

        if chunks is not None:
            options.update({"chunks": chunks})

        # If the metadata resolver has already resolved the overview level,
        # trying to open zoom levels here will result in an error.
        # Better would be to separate uriresolver and driver: https://github.com/Deltares/hydromt/issues/1023
        # Then we can implement looking for a overview level in the driver.
        def _open() -> Union[xr.DataArray, xr.Dataset]:
            try:
                return open_mfraster(uris, mosaic=mosaic, **options)
            except rasterio.errors.RasterioIOError as e:
                if "Cannot open overview level" in str(e):
                    options.pop("overview_level")
                    return open_mfraster(uris, mosaic=mosaic, **options)
                else:
                    raise

        # rasterio uses specific environment variable for s3 access.
        try:
            anon: str = self.filesystem.anon
        except AttributeError:
            anon: str = ""

        if anon:
            with temp_env(**{"AWS_NO_SIGN_REQUEST": "true"}):
                ds = _open()
        else:
            ds = _open()

        # Mosaic's can mess up the chunking, which can error during writing
        # Or maybe setting
        chunks = options.get("chunks")
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

    def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> str:
        """Write out a RasterDataset using rasterio."""
        raise NotImplementedError()

    @staticmethod
    def _get_zoom_levels_and_crs(uri: str) -> Tuple[Dict[int, float], int]:
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
