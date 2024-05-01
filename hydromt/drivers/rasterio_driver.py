"""Driver using rasterio for RasterDataset."""
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import rasterio
import xarray as xr
from pyproj import CRS

from hydromt import io
from hydromt._typing import Geom, StrPath, TimeRange, ZoomLevel
from hydromt._typing.error import NoDataStrategy
from hydromt.config import SETTINGS
from hydromt.data_adapter.caching import cache_vrt_tiles
from hydromt.drivers import RasterDatasetDriver
from hydromt.gis.utils import cellres

logger: Logger = getLogger(__name__)


class RasterioDriver(RasterDatasetDriver):
    """Driver using rasterio for RasterDataset."""

    def read_data(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[ZoomLevel] = None,
        logger: Logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> xr.Dataset:
        """Read data using rasterio."""
        # build up kwargs for open_raster
        kwargs: Dict[str, Any] = {}

        # get source-specific options
        cache_root: str = str(
            self.options.get("cache_root"),
        )

        if cache_root is not None and all([uri.endswith(".vrt") for uri in uris]):
            cache_dir = Path(cache_root) / self.options.get(
                "cache_dir", SETTINGS.cache_dir
            )
            fns_cached = []
            for uri in uris:
                fn1 = cache_vrt_tiles(
                    uri, geom=mask, cache_dir=cache_dir, logger=logger
                )
                fns_cached.append(fn1)
            fns = fns_cached

        # NoData part should be done in DataAdapter.
        # if np.issubdtype(type(self.nodata), np.number):
        #     kwargs.update(nodata=self.nodata)
        if zoom_level is not None and "{zoom_level}" not in uri:
            zls_dict, crs = self._get_zoom_levels_and_crs(fns[0], logger=logger)
            zoom_level = self._parse_zoom_level(
                zoom_level, mask, zls_dict, crs, logger=logger
            )
            if isinstance(zoom_level, int) and zoom_level > 0:
                # NOTE: overview levels start at zoom_level 1, see _get_zoom_levels_and_crs
                kwargs.update(overview_level=zoom_level - 1)
        ds = io.open_mfraster(fns, logger=logger, **kwargs)
        return ds

    def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
        """Write out a RasterDataset using rasterio."""
        pass

    def _get_zoom_levels_and_crs(
        self, fn: Optional[StrPath] = None, logger=logger
    ) -> Tuple[int, int]:
        """Get zoom levels and crs from adapter or detect from tif file if missing."""
        if self.zoom_levels is not None and self.crs is not None:
            return self.zoom_levels, self.crs
        zoom_levels = {}
        crs = None
        if fn is None:
            fn = self.path
        try:
            with rasterio.open(fn) as src:
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
        self.zoom_levels = zoom_levels
        if self.crs is None:
            self.crs = crs
        return zoom_levels, crs

    def _parse_zoom_level(
        self,
        zoom_level: Union[int, Tuple[Union[int, float], str]],
        geom: Optional[Geom] = None,
        zls_dict: Optional[Dict[int, float]] = None,
        dst_crs: Optional[CRS] = None,
        logger=logger,
    ) -> Optional[int]:
        """Return overview level of data corresponding to zoom level.

        Parameters
        ----------
        zoom_level: int or tuple
            overview level or tuple with resolution and unit
        geom: gpd.GeoSeries, optional
            geometry to determine res if zoom_level or source in degree
        zls_dict: dict, optional
            dictionary with overview levels and corresponding resolution
        dst_crs: pyproj.CRS, optional
            destination crs to determine res if zoom_level tuple is provided
            with different unit than dst_crs
        """
        # check zoom level
        zls_dict = self.zoom_levels if zls_dict is None else zls_dict
        dst_crs = self.crs if dst_crs is None else dst_crs
        if zls_dict is None or len(zls_dict) == 0 or zoom_level is None:
            return None
        elif isinstance(zoom_level, int):
            if zoom_level not in zls_dict:
                raise ValueError(
                    f"Zoom level {zoom_level} not defined." f"Select from {zls_dict}."
                )
            zl = zoom_level
            dst_res = zls_dict[zoom_level]
        elif (
            isinstance(zoom_level, tuple)
            and isinstance(zoom_level[0], (int, float))
            and isinstance(zoom_level[1], str)
            and len(zoom_level) == 2
            and dst_crs is not None
        ):
            src_res, src_res_unit = zoom_level
            # convert res if different unit than crs
            dst_crs = CRS.from_user_input(dst_crs)
            dst_crs_unit = dst_crs.axis_info[0].unit_name
            dst_res = src_res
            if dst_crs_unit != src_res_unit:
                known_units = ["degree", "metre", "US survey foot", "meter", "foot"]
                if src_res_unit not in known_units:
                    raise TypeError(
                        f"zoom_level unit {src_res_unit} not understood;"
                        f" should be one of {known_units}"
                    )
                if dst_crs_unit not in known_units:
                    raise NotImplementedError(
                        f"no conversion available for {src_res_unit} to {dst_crs_unit}"
                    )
                conversions = {
                    "foot": 0.3048,
                    "metre": 1,  # official pyproj units
                    "US survey foot": 0.3048,  # official pyproj units
                }  # to meter
                if src_res_unit == "degree" or dst_crs_unit == "degree":
                    lat = 0
                    if geom is not None:
                        lat = geom.to_crs(4326).centroid.y.item()
                    conversions["degree"] = cellres(lat=lat)[1]
                fsrc = conversions.get(src_res_unit, 1)
                fdst = conversions.get(dst_crs_unit, 1)
                dst_res = src_res * fsrc / fdst
            # find nearest zoom level
            res = list(zls_dict.values())[0] / 2
            zls = list(zls_dict.keys())
            smaller = [x < (dst_res + res * 0.01) for x in zls_dict.values()]
            zl = zls[-1] if all(smaller) else zls[max(smaller.index(False) - 1, 0)]
        elif dst_crs is None:
            raise ValueError("No CRS defined, hence no zoom level can be determined.")
        else:
            raise TypeError(f"zoom_level not understood: {type(zoom_level)}")
        logger.debug(f"Using zoom level {zl} ({dst_res:.2f})")
        return zl
