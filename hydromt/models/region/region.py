"""Model Region class."""

from logging import getLogger
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast
from weakref import ReferenceType, ref

from geopandas import GeoDataFrame, gpd
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from shapely import box

from hydromt._typing.type_def import StrPath
from hydromt.models.region._utils import _parse_region
from hydromt.models.root import ModelRoot
from hydromt.workflows.basin_mask import get_basin_geometry

if TYPE_CHECKING:
    from hydromt.models import Model

logger = getLogger(__name__)


class ModelRegion:
    """Define the model region."""

    def __init__(
        self,
        model: "Model",
    ) -> None:
        self.model_ref: ReferenceType["Model"] = ref(model)
        self._data: Optional[GeoDataFrame] = None
        self.logger = model.logger

    def create(
        self,
        region: dict,
        hydrography_fn: str = "merit_hydro",
        basin_index_fn: str = "merit_hydro_index",
    ) -> Dict[str, Any]:
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
        data_catalog = cast("Model", self.model_ref()).data_catalog
        kind, region = _parse_region(
            region, data_catalog=data_catalog, logger=self.logger
        )
        if kind in ["basin", "subbasin", "interbasin"]:
            # retrieve global hydrography data (lazy!)
            ds_org = data_catalog.get_rasterdataset(hydrography_fn)
            if "bounds" not in region:
                region.update(basin_index=data_catalog.get_source(basin_index_fn))
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
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
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

        self._data = geom
        # This setup method returns region so that it can be wrapped for models which
        # require more information, e.g. grid RasterDataArray or xy coordinates.
        return region

    # def _rasterdataset_to_geom(self, raster_dataset) -> GeoDataFrame:
    # crs = raster_dataset.raster.crs
    # coord_index = cast(pd.Index, raster_dataset.raster.coords.to_index())
    # dims_max = cast(np.ndarray, coord_index.max())
    # dims_min = cast(np.ndarray, coord_index.min())

    # # in raster datasets it is guaranteed that y_dim is penultimate dim and x_dim is last dim
    # geom: GeoDataFrame = GeoDataFrame(
    #     geometry=[
    #         box(
    #             xmin=dims_min[-1],
    #             ymin=dims_min[-2],
    #             xmax=dims_max[-1],
    #             ymax=dims_max[-2],
    #         )
    #     ],
    #     crs=crs,
    # )
    # return geom

    def set(self, data: GeoDataFrame, kind: str = "user"):
        """Set the model region based on provided GeoDataFrame."""
        # if nothing is provided, record that the region was set by the user
        self._kind = kind
        self._data = data

    @property
    def total_bounds(self):
        """Return the total bound sof the model region."""
        return self.data.total_bounds

    @property
    def data(self) -> GeoDataFrame:
        """Provide access to the underlying data of the model region."""
        if self._data is None:
            # cast is necessary because technically the model could have been
            # dealocated, but it shouldn't be in this case.
            root: Optional[ModelRoot] = cast("Model", self.model_ref()).root

            # cannot read geom files for purely in memory models
            if root is None:
                raise ValueError("Root was not set, cannot read region file")
            else:
                self.read("region.geojson")

        return cast(GeoDataFrame, self._data)

    @property
    def crs(self) -> CRS:
        """Provide access to the CRS of the model region."""
        return self.data.crs

    def read(
        self,
        rel_path: StrPath = Path("region.geojson"),
        **read_kwargs,
    ):
        """Read the model region from a file on disk."""
        if self._data is None:
            model = cast("Model", self.model_ref())
            root: Optional[ModelRoot] = model.root
            if root.mode.is_reading_mode():
                # cannot read geom files for purely in memory models
                if root is None:
                    raise ValueError("Root was not set, cannot read region file")
                else:
                    self._data = cast(
                        GeoDataFrame,
                        gpd.read_file(join(root.path, rel_path), **read_kwargs),
                    )
                    self._kind = "geom"
            else:
                raise ValueError(
                    "Cannot read while not in read mode either add data or rerun in read mode."
                )

    def write(
        self,
        rel_path: StrPath = Path("region.geojson"),
        to_wgs84=False,
        **write_kwargs,
    ):
        """Write the model region to a file."""
        model = cast("Model", self.model_ref())
        root: Optional[ModelRoot] = model.root

        # cannot read geom files for purely in memory models
        if root is None:
            raise ValueError("Root was not set, cannot read region file")
        if root.mode.is_reading_mode():
            self.read()

        if to_wgs84:
            self._data = self.data.to_crs(4326)

        self.data.to_file(join(root.path, rel_path), **write_kwargs)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ModelRegion):
            return False
        else:
            try:
                assert_geodataframe_equal(self.data, __value.data)
                return True
            except AssertionError:
                return False
