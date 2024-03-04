"""Model Region class."""

from logging import Logger, getLogger
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast
from weakref import ReferenceType, ref

from geopandas import GeoDataFrame, gpd
from pyproj import CRS
from shapely import box
from xugrid import UgridDataArrayAccessor

from hydromt._typing.model_mode import ModelMode
from hydromt._typing.type_def import StrPath
from hydromt.data_catalog import DataCatalog
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
        logger: Logger = logger,
    ) -> None:
        self.model_ref: ReferenceType["Model"] = ref(model)
        self._data: Optional[GeoDataFrame] = None

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
        hydrography_fn : str
            Name of data source for hydrography data.
            FIXME describe data requirements
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
        kind, region = _parse_region(
            region, data_catalog=self.data_catalog, logger=self.logger
        )
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

        self.set_geoms(geom, name="region")

        # This setup method returns region so that it can be wrapped for models which
        # require more information, e.g. grid RasterDataArray or xy coordinates.
        return region

    def create(
        self, region_dict: Dict[str, Any], catalog: Optional[DataCatalog] = None
    ):
        """Calculate the actual model region."""
        self._kind, data = _parse_region(region_dict, catalog, logger)
        if self._kind == "mesh":
            self._data = cast(UgridDataArrayAccessor, data["mesh"]).to_geodataframe()

        elif self._kind == "bbox":
            self._data = gpd.GeoDataFrame(
                geometry=[box(*data["bbox"])], crs=CRS.from_epsg(4326)
            )
        elif self._kind == "geom":
            self._data = data["geom"]
        elif self._kind == "grid":
            raise NotImplementedError("TODO")
        elif self._kind in ["basin", "subbasin", "interbasin"]:
            raise NotImplementedError("TODO")
        else:
            raise ValueError(f"Could not understand kind {self._kind}")

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
        model_mode: ModelMode = ModelMode.READ,
        **read_kwargs,
    ):
        """Read the model region from a file on disk."""
        if self._data is None:
            if model_mode.is_reading_mode():
                root: Optional[ModelRoot] = cast("Model", self.model_ref()).root

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
                raise ValueError("Cannot read while not in read mode")

    def write(
        self,
        rel_path: Path = Path("region.geojson"),
        model_mode: ModelMode = ModelMode.WRITE,
        **write_kwargs,
    ):
        """Write the model region to a file."""
        if model_mode.is_writing_mode():
            root: Optional[ModelRoot] = cast("Model", self.model_ref()).root

            # cannot read geom files for purely in memory models
            if root is None:
                raise ValueError("Root was not set, cannot read region file")
            else:
                self.read()
            self.data.to_file(join(root.path, rel_path), **write_kwargs)
