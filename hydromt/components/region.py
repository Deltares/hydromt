"""Model Region class."""

from logging import getLogger
from os import makedirs
from os.path import basename, exists, isdir, isfile, join
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu
from geopandas import GeoDataFrame, GeoSeries
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from shapely import box

from hydromt import _compat, hydromt_step
from hydromt._typing.type_def import StrPath
from hydromt.components.base import ModelComponent
from hydromt.data_catalog import DataCatalog
from hydromt.gis import utils as gis_utils
from hydromt.plugins import PLUGINS
from hydromt.workflows.basin_mask import get_basin_geometry

if TYPE_CHECKING:
    from hydromt.models import Model


logger = getLogger(__name__)

DEFAULT_REGION_FILE_PATH = "region.geojson"


class ModelRegionComponent(ModelComponent):
    """Define the model region."""

    def __init__(
        self,
        model: "Model",
    ) -> None:
        super().__init__(model)
        self._data: Optional[GeoDataFrame] = None

    @hydromt_step
    def create(
        self,
        *,
        region: dict,
        crs: Optional[int] = None,
        hydrography_fn: str = "merit_hydro",
        basin_index_fn: str = "merit_hydro_index",
    ) -> None:
        """Check and return parsed region arguments.

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest.

            For an exact clip of the region:

            * {'bbox': [xmin, ymin, xmax, ymax]}

            * {'geom': /path/to/polygon_geometry}

            For a region based of another models grid:

            * {'<model_name>': root}

            For a region based of the grid of a raster file:

            * {'grid': /path/to/raster}

            For a region based on a mesh grid of a mesh file:

            * {'mesh': /path/to/mesh}

            Entire basin can be defined based on an ID, one or multiple point location
            (x, y), or a region of interest (bounding box or geometry) for which the
            basin IDs are looked up. The basins withint the area of interest can be further
            filtered to only include basins with their outlet within the area of interest
            ('outlets': true) of stream threshold arguments (e.g.: 'uparea': 1000).

            Common use-cases include:

            * {'basin': ID}

            * {'basin': [ID1, ID2, ..]}

            * {'basin': [x, y]}

            * {'basin': [[x1, x2, ..], [y1, y2, ..]]}

            * {'basin': /path/to/point_geometry}

            * {'basin': [xmin, ymin, xmax, ymax]}

            * {'basin': [xmin, ymin, xmax, ymax], 'outlets': true}

            * {'basin': [xmin, ymin, xmax, ymax], '<variable>': threshold}

            Subbasins are defined by its outlet locations and include all area upstream
            from these points. The outlet locations can be passed as xy coordinate pairs,
            but also derived from the most downstream cell(s) within a area of interest
            defined by a bounding box or geometry, optionally refined by stream threshold
            arguments.

            The method can be speed up by providing an additional ``bounds`` argument which
            should contain all upstream cell. If cells upstream of the subbasin are not
            within the provide bounds a warning will be raised. Common use-cases include:

            * {'subbasin': [x, y], '<variable>': threshold}

            * {
                'subbasin': [[x1, x2, ..], [y1, y2, ..]],
                '<variable>': threshold, 'bounds': [xmin, ymin, xmax, ymax]
                }

            * {'subbasin': /path/to/point_geometry, '<variable>': threshold}

            * {'subbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}

            * {'subbasin': /path/to/polygon_geometry, '<variable>': threshold}

            Interbasins are similar to subbasins but are bounded by a bounding box or
            geometry and do not include all upstream area. Common use-cases include:

            * {'interbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}

            * {'interbasin': [xmin, ymin, xmax, ymax], 'xy': [x, y]}

            * {'interbasin': /path/to/polygon_geometry, 'outlets': true}
        logger:
            The logger to use.
        crs: int, optional
            EPSG code of the model or "utm" to let hydromt find the closest projected

        Returns
        -------
        kind : {'basin', 'subbasin', 'interbasin', 'geom', 'bbox', 'grid'}
            region kind
        kwargs : dict
            parsed region json
        """
        if self.data is not None:
            self._logger.warn("Model region already initialized. Skipping creation.")
            return

        kind, region = _parse_region(
            region, data_catalog=self._data_catalog, logger=self._logger
        )
        if kind in ["basin", "subbasin", "interbasin"]:
            # retrieve global hydrography data (lazy!)
            ds_org = self._data_catalog.get_rasterdataset(hydrography_fn)
            if "bounds" not in region:
                region.update(basin_index=self._data_catalog.get_source(basin_index_fn))
            # get basin geometry
            geom, xy = get_basin_geometry(
                ds=ds_org,
                kind=kind,
                logger=self._logger,
                **region,
            )
            region.update(xy=xy)
            # get ds_hyd again but clipped to geom, one variable is enough
            da_hyd = self._data_catalog.get_rasterdataset(
                hydrography_fn, geom=geom, variables=["flwdir"]
            )
            assert da_hyd is not None
            if geom.crs != da_hyd.raster.crs:
                crs = da_hyd.raster.crs
                geom = geom.to_crs(crs)
        elif "bbox" in region:
            bbox = region["bbox"]
            # TODO: Use the crs from the parameters directly?
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
            if crs is not None:
                crs = gis_utils.parse_crs(crs, bbox=geom.total_bounds)
                geom = geom.to_crs(crs)
        elif "geom" in region:
            geom = region["geom"]
            # TODO: What if the crs is defined in the parameters? grid.py also used to raise an error.
            if geom.crs is None:
                raise ValueError('Model region "geom" has no CRS')
            if crs is not None:
                crs = gis_utils.parse_crs(crs, bbox=geom.total_bounds)
                geom = geom.to_crs(crs)
        elif "grid" in region:  # Grid specific - should be removed in the future
            geom = region["grid"].raster.box
            # TODO: Update crs with argument from function?
            if crs is not None:
                self._logger.warning(
                    f"For region kind 'grid', the grid's crs is used and not user-defined crs {crs}"
                )
        elif "model" in region:
            geom = region["model"].region
        else:
            raise ValueError(f"model region argument not understood: {region}")

        self.set(geom, kind)

    def set(self, data: GeoDataFrame, kind: str = "geom") -> None:
        """Set the model region based on provided GeoDataFrame."""
        # if nothing is provided, record that the region was set by the user
        self.kind = kind
        if self._data is not None:
            self._logger.info("Updating region geometry.")

        if isinstance(data, GeoSeries):
            self._data = GeoDataFrame(data)
        elif isinstance(data, GeoDataFrame):
            self._data = data
        else:
            raise ValueError("Only GeoSeries or GeoDataFrame can be used as region.")

    @property
    def bounds(self):
        """Return the total bound sof the model region."""
        if self.data is not None:
            return self.data.total_bounds
        else:
            raise ValueError("Could not read or construct region to read bounds from")

    @property
    def data(self) -> Optional[GeoDataFrame]:
        """Provide access to the underlying GeoDataFrame data of the model region."""
        if self._data is None and self._root.is_reading_mode():
            self.read()

        return self._data

    @property
    def crs(self) -> CRS:
        """Provide access to the CRS of the model region."""
        if self.data is not None:
            return self.data.crs
        else:
            raise ValueError("Could not read or construct region to read crs from")

    @hydromt_step
    def read(
        self,
        rel_path: StrPath = Path(DEFAULT_REGION_FILE_PATH),
        **read_kwargs,
    ):
        """Read the model region from a file on disk."""
        self._root._assert_read_mode()
        # cannot read geom files for purely in memory models
        self._logger.debug(f"Reading model file {rel_path}.")
        self._data = cast(
            GeoDataFrame,
            gpd.read_file(join(self._root.path, rel_path), **read_kwargs),
        )
        self.kind = "geom"

    @hydromt_step
    def write(
        self,
        rel_path: StrPath = Path(DEFAULT_REGION_FILE_PATH),
        to_wgs84=False,
        **write_kwargs,
    ):
        """Write the model region to a file."""
        self._root._assert_write_mode()
        write_path = join(self._root.path, rel_path)

        if exists(write_path) and not self._root.is_override_mode():
            raise OSError(
                f"Model dir already exists and cannot be overwritten: {write_path}"
            )
        base_name = basename(write_path)
        if not exists(base_name):
            makedirs(base_name, exist_ok=True)

        if self.data is None:
            self._logger.info("No region data found. skipping writing...")
        else:
            self._logger.info(f"writing region data to {write_path}")
            gdf = self.data.copy()

            if to_wgs84 and (
                write_kwargs.get("driver") == "GeoJSON"
                or str(rel_path).lower().endswith(".geojson")
            ):
                gdf = gdf.to_crs(4326)

            gdf.to_file(write_path, **write_kwargs)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ModelRegionComponent):
            return False
        else:
            try:
                assert_geodataframe_equal(self.data, __value.data)
                return True
            except AssertionError:
                return False


def _parse_region(
    region, data_catalog=None, logger=logger
) -> Tuple[str, Dict[str, Any]]:
    if data_catalog is None:
        data_catalog = DataCatalog()
    kwargs = region.copy()
    # NOTE: the order is important to prioritize the arguments
    options = {
        "basin": ["basid", "geom", "bbox", "xy"],
        "subbasin": ["geom", "bbox", "xy"],
        "interbasin": ["geom", "bbox", "xy"],
        "geom": ["geom"],
        "bbox": ["bbox"],
        "grid": ["RasterDataArray"],
        "mesh": ["UgridDataArray"],
    }
    kind = next(iter(kwargs))  # first key of region
    value0 = kwargs.pop(kind)

    if kind == "grid":
        kwargs = {"grid": data_catalog.get_rasterdataset(value0, driver_kwargs=kwargs)}
    elif kind in PLUGINS.model_plugins:
        model_class = PLUGINS.model_plugins[kind]
        kwargs = dict(mod=model_class(root=value0, mode="r", logger=logger))
        kind = "model"
    elif kind == "mesh":
        if _compat.HAS_XUGRID:
            if isinstance(value0, (str, Path)) and isfile(value0):
                kwarg = dict(mesh=xu.open_dataset(value0))
            elif isinstance(value0, (xu.UgridDataset, xu.UgridDataArray)):
                kwarg = dict(mesh=value0)
            elif isinstance(value0, (xu.Ugrid1d, xu.Ugrid2d)):
                kwarg = dict(
                    mesh=xu.UgridDataset(value0.to_dataset(optional_attributes=True))
                )
            else:
                raise ValueError(
                    f"Unrecognised type {type(value0)}."
                    "Should be a path, data catalog key or xugrid object."
                )
            kwargs.update(kwarg)
        else:
            raise ImportError("xugrid is required to read mesh files.")
    elif kind not in options:
        k_lst = '", "'.join(list(options.keys()))
        raise ValueError(f'Region key "{kind}" not understood, select from "{k_lst}"')
    else:
        kwarg = _parse_region_value(value0, data_catalog=data_catalog)
        if len(kwarg) == 0 or next(iter(kwarg)) not in options[kind]:
            v_lst = '", "'.join(list(options[kind]))
            raise ValueError(
                f'Region value "{value0}" for kind={kind} not understood, '
                f'provide one of "{v_lst}"'
            )
        kwargs.update(kwarg)
    kwargs_str = dict()
    for k, v in kwargs.items():
        if isinstance(v, gpd.GeoDataFrame):
            v = f"GeoDataFrame {v.total_bounds} (crs = {v.crs})"
        elif isinstance(v, xr.DataArray):
            v = f"DataArray {v.raster.bounds} (crs = {v.raster.crs})"
        kwargs_str.update({k: v})
    logger.debug(f"Parsed region (kind={kind}): {str(kwargs_str)}")
    return kind, kwargs


def _parse_region_value(value, data_catalog):
    kwarg = {}
    if isinstance(value, np.ndarray):
        value = value.tolist()  # array to list

    if isinstance(value, list):
        if np.all([isinstance(p0, int) and abs(p0) > 180 for p0 in value]):  # all int
            kwarg = dict(basid=value)
        elif len(value) == 4:  # 4 floats
            kwarg = dict(bbox=value)
        elif len(value) == 2:  # 2 floats
            kwarg = dict(xy=value)
    elif isinstance(value, tuple) and len(value) == 2:  # tuple of x and y coords
        kwarg = dict(xy=value)
    elif isinstance(value, int):  # single int
        kwarg = dict(basid=value)
    elif isinstance(value, (str, Path)) and isdir(value):
        kwarg = dict(root=value)
    elif isinstance(value, (str, Path)):
        geom = data_catalog.get_geodataframe(value)
        kwarg = dict(geom=geom)
    elif isinstance(value, gpd.GeoDataFrame):  # geometry
        kwarg = dict(geom=value)
    else:
        raise ValueError(f"Region value {value} not understood.")

    if "geom" in kwarg and np.all(kwarg["geom"].geometry.type == "Point"):
        xy = (
            kwarg["geom"].geometry.x.values,
            kwarg["geom"].geometry.y.values,
        )
        kwarg = dict(xy=xy)
    return kwarg
