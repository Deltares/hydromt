"""Model Region class."""

from abc import ABC
from logging import getLogger
from os import makedirs
from os.path import basename, exists, isdir, isfile, join
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast

import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu
from geopandas import GeoDataFrame, GeoSeries
from pyproj import CRS
from shapely import box

from hydromt import _compat
from hydromt._typing.type_def import StrPath
from hydromt.components.base import ModelComponent
from hydromt.data_catalog import DataCatalog
from hydromt.gis import utils as gis_utils
from hydromt.plugins import PLUGINS
from hydromt.workflows.basin_mask import get_basin_geometry

if TYPE_CHECKING:
    from hydromt.models import Model


logger = getLogger(__name__)


class SpatialModelComponent(ModelComponent, ABC):
    """Define the model region."""

    DEFAULT_REGION_FILENAME = "region.geojson"
    DEFAULT_HYDROGRAPHY_FILENAME = "merit_hydro"
    DEFAULT_BASIN_INDEX_FILENAME = "merit_hydro_index"

    def __init__(
        self,
        model: "Model",
        region_component: Optional[str] = None,
        region_filename: StrPath = DEFAULT_REGION_FILENAME,
    ) -> None:
        super().__init__(model)
        self.__filename: StrPath = region_filename
        self.__data: Optional[GeoDataFrame] = None
        self._region_component = region_component

    def create_region(
        self,
        *,
        region: dict,
        hydrography_fn: str = DEFAULT_HYDROGRAPHY_FILENAME,
        basin_index_fn: str = DEFAULT_BASIN_INDEX_FILENAME,
        crs: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """Create the region and set it on the region attribute.

        This function should be called from within the `create` function of the component inheriting from this class.

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest.

            For an exact clip of the region:

            * {'bbox': [xmin, ymin, xmax, ymax]}

            * {'geom': /path/to/polygon_geometry}

            For a region based of another models grid:

            * {'<model_name>': root}

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
        """
        region_from_reference = self._get_region_from_reference()
        if region_from_reference is not None:
            return region_from_reference

        geom = self.parse_region(
            region,
            basin_index_fn=basin_index_fn,
            hydrography_fn=hydrography_fn,
            crs=crs,
        )
        self.set_region(geom)
        return geom

    def set_region(self, data: Union[GeoDataFrame, GeoSeries]) -> None:
        """Set the model region based on provided GeoDataFrame."""
        # if nothing is provided, record that the region was set by the user
        if self.__data is not None:
            self._logger.info("Updating region geometry.")

        if isinstance(data, GeoSeries):
            self.__data = GeoDataFrame(data)
        elif isinstance(data, GeoDataFrame):
            self.__data = data
        else:
            raise ValueError("Only GeoSeries or GeoDataFrame can be used as region.")

    @property
    def bounds(self):
        """Return the total bounds of the model region."""
        if self.region is not None:
            return self.region.total_bounds
        return None

    @property
    def region(self) -> Optional[GeoDataFrame]:
        """Provide access to the underlying GeoDataFrame data of the model region."""
        region_from_reference = self._get_region_from_reference()
        if region_from_reference is not None:
            return region_from_reference

        if self.__data is None and self._root.is_reading_mode():
            self.read_region()
        return self.__data

    @property
    def crs(self) -> Optional[CRS]:
        """Provide access to the CRS of the model region."""
        if self.region is not None:
            return self.region.crs
        return None

    def read_region(
        self,
        filename: Optional[StrPath] = None,
        **read_kwargs,
    ) -> None:
        """Read the model region from a file on disk. Sets it on the region attribute.

        This function should be called from within the `read` function of the component inheriting from this class.
        """
        self._root._assert_read_mode()
        if self._region_component is not None:
            self._logger.info(
                "Region is a reference to another component. Skipping reading..."
            )
            return

        filename = filename or self.__filename
        # cannot read geom files for purely in memory models
        self._logger.debug(f"Reading model file {filename}.")
        self.set_region(gpd.read_file(join(self._root.path, filename), **read_kwargs))

    def write_region(
        self,
        filename: Optional[StrPath] = None,
        *,
        to_wgs84=False,
        **write_kwargs,
    ):
        """Write the model region to a file.

        This function should be called from within the `write` function of the component inheriting from this class.
        """
        self._root._assert_write_mode()
        if self._region_component is not None:
            self._logger.info(
                "Region is a reference to another component. Skipping writing..."
            )
            return

        filename = filename or self.__filename
        write_path = join(self._root.path, filename)

        if exists(write_path) and not self._root.is_override_mode():
            raise OSError(
                f"Model dir already exists and cannot be overwritten: {write_path}"
            )
        base_name = basename(write_path)
        if not exists(base_name):
            makedirs(base_name, exist_ok=True)

        if self.region is None:
            self._logger.info("No region data found. skipping writing...")
        else:
            self._logger.info(f"writing region data to {write_path}")
            gdf = self.region.copy()

            if to_wgs84 and (
                write_kwargs.get("driver") == "GeoJSON"
                or str(filename).lower().endswith(".geojson")
            ):
                gdf = gdf.to_crs(4326)

            gdf.to_file(write_path, **write_kwargs)

    def parse_region(
        self,
        region: dict,
        *,
        hydrography_fn: str,
        basin_index_fn: str,
        crs: Optional[int],
    ) -> gpd.GeoDataFrame:
        """Parse a region dictionary and return a GeoDataFrame.

        This function can be overridden by subclasses to provide custom region parsing.

        Parameters
        ----------
        region : dict
            The region description to be parsed.
            See `create_region` for more information.
        hydrography_fn : str
            The hydrography filename.
        basin_index_fn : str
            The basin index filename.
        crs : Optional[int]
            The target crs to apply to the region after parsing.

        Returns
        -------
        gpd.GeoDataFrame
            The region created from the parsed region dictionary.
        """
        kwargs = region.copy()
        # NOTE: the order is important to prioritize the arguments
        options = {
            "basin": ["basid", "geom", "bbox", "xy"],
            "subbasin": ["geom", "bbox", "xy"],
            "interbasin": ["geom", "bbox", "xy"],
            "geom": ["geom"],
            "bbox": ["bbox"],
            "mesh": ["UgridDataArray"],
        }
        kind = next(iter(kwargs))  # first key of region
        value0 = kwargs.pop(kind)

        if kind in PLUGINS.model_plugins:
            model_class = PLUGINS.model_plugins[kind]
            other_model = model_class(root=value0, mode="r", logger=self._logger)
            geom = other_model.region
        # TODO: Move to MeshComponent
        elif kind == "mesh":
            if _compat.HAS_XUGRID:
                if isinstance(value0, (str, Path)) and isfile(value0):
                    kwarg = dict(mesh=xu.open_dataset(value0))
                elif isinstance(value0, (xu.UgridDataset, xu.UgridDataArray)):
                    kwarg = dict(mesh=value0)
                elif isinstance(value0, (xu.Ugrid1d, xu.Ugrid2d)):
                    kwarg = dict(
                        mesh=xu.UgridDataset(
                            value0.to_dataset(optional_attributes=True)
                        )
                    )
                else:
                    raise ValueError(
                        f"Unrecognized type {type(value0)}."
                        "Should be a path, data catalog key or xugrid object."
                    )
                kwargs.update(kwarg)
            else:
                raise ImportError("xugrid is required to read mesh files.")
        elif kind not in options:
            k_lst = '", "'.join(list(options.keys()))
            raise ValueError(
                f'Region key "{kind}" not understood, select from "{k_lst}"'
            )

        kwarg = _parse_region_value(value0, data_catalog=self._data_catalog)
        if len(kwarg) == 0 or next(iter(kwarg)) not in options[kind]:
            v_lst = '", "'.join(list(options[kind]))
            raise ValueError(
                f'Region value "{value0}" for kind={kind} not understood, '
                f'provide one of "{v_lst}"'
            )
        kwargs.update(kwarg)

        if kind in ["basin", "subbasin", "interbasin"]:
            # retrieve global hydrography data (lazy!)
            ds_org = self._data_catalog.get_rasterdataset(hydrography_fn)
            if "bounds" not in region:
                region.update(basin_index=self._data_catalog.get_source(basin_index_fn))
            # get basin geometry
            geom, _ = get_basin_geometry(
                ds=ds_org,
                kind=kind,
                logger=self._logger,
                **region,
            )
        elif "bbox" in region:
            bbox = region["bbox"]
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
        elif "geom" in region:
            geom = region["geom"]
            if geom.crs is None:
                raise ValueError('Model region "geom" has no CRS')

        if crs is not None and geom is not None:
            crs = gis_utils.parse_crs(crs, bbox=geom.total_bounds)
            geom = geom.to_crs(crs)

        # TODO: This won't be called when subclassed. Should this move to _create_region?
        kwargs_str = dict()
        for k, v in kwargs.items():
            if isinstance(v, gpd.GeoDataFrame):
                v = f"GeoDataFrame {v.total_bounds} (crs = {v.crs})"
            elif isinstance(v, xr.DataArray):
                v = f"DataArray {v.raster.bounds} (crs = {v.raster.crs})"
            kwargs_str.update({k: v})
        self._logger.debug(f"Parsed region (kind={kind}): {str(kwargs_str)}")

        return geom

    def test_equal(self, other: ModelComponent) -> Tuple[bool, Dict[str, str]]:
        """Test if two components are equal.

        Parameters
        ----------
        other : ModelComponent
            The component to compare against.

        Returns
        -------
        tuple[bool, Dict[str, str]]
            True if the components are equal, and a dict with the associated errors per property checked.
        """
        eq, errors = super().test_equal(other)
        if not eq:
            return eq, errors
        other_region = cast(SpatialModelComponent, other)

        try:
            gpd.testing.assert_geodataframe_equal(
                self.region,
                other_region.region,
                check_like=True,
                check_less_precise=True,
            )
        except AssertionError as e:
            errors["data"] = str(e)

        return len(errors) == 0, errors

    def _get_region_from_reference(self) -> Optional[gpd.GeoDataFrame]:
        if self._region_component is not None:
            region_component = self._model.get_component(
                self._region_component, SpatialModelComponent
            )
            if region_component is None:
                raise ValueError(
                    f"Unable to find the referenced region component: '{self._region_component}'"
                )
            if region_component.region is None:
                raise ValueError(
                    f"Unable to get region from the referenced region component: '{self._region_component}'"
                )
            return region_component.region
        return None


# TODO: Remove when migrating MeshComponent
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
