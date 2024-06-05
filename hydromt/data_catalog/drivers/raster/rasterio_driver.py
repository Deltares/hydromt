"""Driver using rasterio for RasterDataset."""

from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from hydromt._typing import (
    Geom,
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
    ZoomLevel,
)
from hydromt._typing.error import NoDataStrategy
from hydromt._utils import _cache_vrt_tiles, _strip_scheme
from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs
from hydromt.config import SETTINGS
from hydromt.data_catalog.drivers import RasterDatasetDriver
from hydromt.io.readers import open_mfraster

logger: Logger = getLogger(__name__)


class RasterioDriver(RasterDatasetDriver):
    """Driver using rasterio for RasterDataset."""

    name = "rasterio"

    def read_data(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        time_range: Optional[TimeRange] = None,
        variables: Optional[Variables] = None,
        zoom_level: Optional[ZoomLevel] = None,
        metadata: Optional[SourceMetadata] = None,
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> xr.Dataset:
        """Read data using rasterio."""
        if metadata is None:
            metadata = SourceMetadata()
        # build up kwargs for open_raster
        _warn_on_unused_kwargs(
            self.__class__.__name__,
            {"time_range": time_range, "zoom_level": zoom_level},
            logger=logger,
        )
        kwargs: Dict[str, Any] = {}

        # get source-specific options
        cache_root: str = str(
            self.options.get("cache_root", SETTINGS.cache_root),
        )

        if cache_root is not None and all([uri.endswith(".vrt") for uri in uris]):
            cache_dir = Path(cache_root) / self.options.get(
                "cache_dir",
                Path(
                    _strip_scheme(uris[0])
                ).stem,  # default to first uri without extension
            )
            uris_cached = []
            for uri in uris:
                cached_uri: str = _cache_vrt_tiles(
                    uri, geom=mask, cache_dir=cache_dir, logger=logger
                )
                uris_cached.append(cached_uri)
            uris = uris_cached

        # NoData part should be done in DataAdapter.
        if np.issubdtype(type(metadata.nodata), np.number):
            kwargs.update(nodata=metadata.nodata)
        # TODO: Implement zoom levels in https://github.com/Deltares/hydromt/issues/875
        # if zoom_level is not None and "{zoom_level}" not in uri:
        #     zls_dict, crs = self._get_zoom_levels_and_crs(uris[0], logger=logger)
        #     zoom_level = self._parse_zoom_level(
        #         zoom_level, mask, zls_dict, crs, logger=logger
        #     )
        #     if isinstance(zoom_level, int) and zoom_level > 0:
        #         # NOTE: overview levels start at zoom_level 1, see _get_zoom_levels_and_crs
        #         kwargs.update(overview_level=zoom_level - 1)

        if mask is not None:
            kwargs.update({"mosaic_kwargs": {"mask": mask}})
        ds = open_mfraster(uris, logger=logger, **kwargs)
        # rename ds with single band if single variable is requested
        if variables is not None and len(variables) == 1 and len(ds.data_vars) == 1:
            ds = ds.rename({list(ds.data_vars.keys())[0]: list(variables)[0]})
        return ds

    def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
        """Write out a RasterDataset using rasterio."""
        raise NotImplementedError()
