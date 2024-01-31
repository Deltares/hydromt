"""A class to handle all functionalities to do with model regions."""

from logging import Logger, getLogger
from os.path import isdir, isfile
from pathlib import Path
from typing import Optional

import numpy as np
from geopandas import GeoDataFrame
from pyproj import CRS
from xarray import DataArray

from hydromt._compat import HAS_XUGRID
from hydromt.typing import Bbox
from hydromt.workflows import get_basin_geometry

if HAS_XUGRID:
    import xugrid as xu

logger = getLogger(__name__)


class ModelRegion:
    """A class to handle all operations to do with the model region."""

    def __init__(
        self,
        bbox: Optional[Bbox] = None,
        geoms: Optional[GeoDataFrame] = None,
        crs: Optional[CRS] = None,
        logger: Logger = logger,
    ):
        self.logger = logger
        self.set(bbox=bbox, geoms=geoms, crs=crs)

    def set(
        self,
        bbox: Optional[Bbox] = None,
        geoms: Optional[GeoDataFrame] = None,
        basin_index: Optional[GeoDataFrame] = None,
        hydrography: Optional[GeoDataFrame] = None,
        crs: Optional[CRS] = None,
    ):
        """Set the the model region."""
        if bbox is not None and geoms is not None:
            raise ValueError("Only supply one of bbox and geoms.")

        self.bbox = bbox
        self.geoms = geoms
        self.crs = crs

    def read(self):
        """Read the model geom from a file."""
        pass

    def write(self):
        """Write the model geom to a file."""
        pass

    def setup_region(
        self,
        region: dict,
        hydrography_fn: str = "merit_hydro",
        basin_index_fn: str = "merit_hydro_index",
    ) -> dict:
        """Set the `region` of interest of the model.

        Adds model layer:

        * **region** geom: region boundary vector

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g.:
            * {'bbox': [xmin, ymin, xmax, ymax]}
            * {'geom': 'path/to/polygon_geometry'}
            * {'basin': [xmin, ymin, xmax, ymax]}
            * {'subbasin': [x, y], '<variable>': threshold}
            For a complete overview of all region options,
            see :py:function:~hydromt.workflows.basin_mask.parse_region
        hydrography_ds : str
            Name of data source for hydrography data.
        basin_index_fn : str
            Name of data source with basin (bounding box) geometries associated with
            the 'basins' layer of `hydrography_fn`. Only required if the `region` is
            based on a (sub)(inter)basins without a 'bounds' argument.

        Returns
        -------
        region: dict
            Parsed region dictionary

        See Also
        --------
        hydromt.workflows.basin_mask.parse_region
        """
        kind, region = self._parse_region(region, logger=self.logger)
        if kind in ["basin", "subbasin", "interbasin"]:
            # retrieve global hydrography data (lazy!)
            ds_org = self.data_catalog.get_rasterdataset(hydrography_fn)
            if "bounds" not in region:
                region.update(basin_index=self.data_catalog.get_source(basin_index_fn))
            # get basin geometry
            geom, xy = get_basin_geometry(
                ds=ds_org,
                kind=kind,
                logger=self.logger,
                **region,
            )
            region.update(xy=xy)
        elif "bbox" in region:
            bbox = region["bbox"]
            geom = GeoDataFrame(geometry=[box(*bbox)], crs=4326)
        elif "geom" in region:
            geom = region["geom"]
            if geom.crs is None:
                raise ValueError('Model region "geom" has no CRS')
        elif "grid" in region:  # Grid specific - should be removed in the future
            geom = region["grid"].raster.box
        elif "model" in region:
            geom = region["model"].region
        else:
            raise ValueError(f"model region argument not understood: {region}")

        # This setup method returns region so that it can be wrapped for models which
        # require more information, e.g. grid RasterDataArray or xy coordinates.
        return region

    def _parse_region(region, logger=logger):
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

        Returns
        -------
        kind : {'basin', 'subbasin', 'interbasin', 'geom', 'bbox', 'grid'}
            region kind
        kwargs : dict
            parsed region json
        """
        kwargs = region.copy()

        # NOTE: the order is important to prioritize the arguments
        options = {
            "basin": ["basid", "geom", "bbox", "xy"],
            "subbasin": ["geom", "bbox", "xy"],
            "interbasin": [
                "geom",
                "bbox",
            ],
            "geom": ["geom"],
            "bbox": ["bbox"],
            "grid": ["RasterDataArray"],
            "mesh": ["UgridDataArray"],
        }
        kind = next(iter(kwargs))  # first key of region
        value = kwargs.pop(kind)
        if kind in MODELS:
            model_class = MODELS.load(kind)
            kwargs = dict(mod=model_class(root=value, mode="r", logger=logger))
            kind = "model"
        elif kind == "grid":
            kwargs = {
                "grid": data_catalog.get_rasterdataset(value, driver_kwargs=kwargs)
            }
        elif kind == "mesh":
            if _compat.HAS_XUGRID:
                if isinstance(value, (str, Path)) and isfile(value):
                    kwarg = dict(mesh=xu.open_dataset(value))
                elif isinstance(value, (xu.UgridDataset, xu.UgridDataArray)):
                    kwarg = dict(mesh=value)
                elif isinstance(value, (xu.Ugrid1d, xu.Ugrid2d)):
                    kwarg = dict(
                        mesh=xu.UgridDataset(value.to_dataset(optional_attributes=True))
                    )
                else:
                    raise ValueError(
                        f"Unrecognised type {type(value)}."
                        "Should be a path, data catalog key or xugrid object."
                    )
                kwargs.update(kwarg)
            else:
                raise ImportError("xugrid is required to read mesh files.")
        elif kind not in options:
            k_lst = '", "'.join(list(options.keys()) + list(MODELS))
            raise ValueError(
                f'Region key "{kind}" not understood, select from "{k_lst}"'
            )
        else:
            kwarg = _parse_region_value(value, data_catalog=data_catalog)
            if len(kwarg) == 0 or next(iter(kwarg)) not in options[kind]:
                v_lst = '", "'.join(list(options[kind]))
                raise ValueError(
                    f'Region value "{value}" for kind={kind} not understood, '
                    f'provide one of "{v_lst}"'
                )
            kwargs.update(kwarg)
        kwargs_str = dict()
        for k, v in kwargs.items():
            if isinstance(v, GeoDataFrame):
                v = f"GeoDataFrame {v.total_bounds} (crs = {v.crs})"
            elif isinstance(v, DataArray):
                v = f"DataArray {v.raster.bounds} (crs = {v.raster.crs})"
            kwargs_str.update({k: v})
        logger.debug(f"Parsed region (kind={kind}): {str(kwargs_str)}")
        return kind, kwargs

    def _parse_region_value(value):
        kwarg = {}
        if isinstance(value, np.ndarray):
            value = value.tolist()  # array to list

        if isinstance(value, list):
            if np.all(
                [isinstance(p0, int) and abs(p0) > 180 for p0 in value]
            ):  # all int
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
        elif isinstance(value, GeoDataFrame):  # geometry
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

        # # merge cli region and res arguments with opt
        # if region is not None:
        #     if self._CLI_ARGS["region"] not in opt:
        #         opt = {self._CLI_ARGS["region"]: {}, **opt}
        #     opt[self._CLI_ARGS["region"]].update(region=region)
