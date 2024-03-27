"""Table component."""

import glob
import os
from os.path import basename, dirname, join
from typing import (
    TYPE_CHECKING,
    Dict,
    Optional,
    Union,
    cast,
)

import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries

from hydromt.components.base import ModelComponent
from hydromt.hydromt_step import hydromt_step

if TYPE_CHECKING:
    from hydromt.models.model import Model


class GeomComponent(ModelComponent):
    """Geom Component."""

    def __init__(
        self,
        model: "Model",
    ):
        """Initialize a GeomComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        """
        self._data: Optional[Dict[str, Union[GeoDataFrame, GeoSeries]]] = None
        super().__init__(model=model)

    @property
    def data(self) -> Dict[str, Union[GeoDataFrame, GeoSeries]]:
        """Model geometries.

        Return dict of geopandas.GeoDataFrame or geopandas.GeoDataSeries
        ..NOTE: previously call staticgeoms.
        """
        if self._data is None:
            self._initialize_geoms()
        assert self._data is not None
        return self._data

    def _initialize_geoms(self, skip_read=False) -> None:
        """Initialize geoms."""
        if self._geoms is None:
            self._geoms = dict()
            if self._root.is_reading_mode() and not skip_read:
                self.read()

    def set(self, geom: Union[GeoDataFrame, GeoSeries], name: str):
        """Add data to the geoms attribute.

        Arguments
        ---------
        geom: geopandas.GeoDataFrame or geopandas.GeoSeries
            New geometry data to add
        name: str
            Geometry name.
        """
        self._initialize_geoms()
        if name in self._geoms:
            self._logger.warning(f"Replacing geom: {name}")
        if hasattr(self, "crs"):
            # Verify if a geom is set to model crs and if not sets geom to model crs
            if self.crs and self.crs != geom.crs:
                geom.to_crs(self.crs.to_epsg(), inplace=True)
        self._geoms[name] = geom

    @hydromt_step
    def read(self, fn: str = "geoms/*.geojson", **kwargs) -> None:
        r"""Read model geometries files at <root>/<fn> and add to geoms property.

        key-word arguments are passed to :py:func:`geopandas.read_file`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may contain wildcards,
            by default ``geoms/\*.nc``
        **kwargs:
            Additional keyword arguments that are passed to the
            `geopandas.read_file` function.
        """
        self._root._assert_read_mode()
        self._initialize_geoms(skip_read=True)
        fns = glob.glob(join(self._root.path, fn))
        for fn in fns:
            name = basename(fn).split(".")[0]
            self._logger.debug(f"Reading model file {name}.")
            geom = cast(GeoDataFrame, gpd.read_file(fn, **kwargs))

            self.set(geom=geom, name=name)

    @hydromt_step
    def write(
        self, fn: str = "geoms/{name}.geojson", to_wgs84: bool = False, **kwargs
    ) -> None:
        r"""Write model geometries to a vector file (by default GeoJSON) at <root>/<fn>.

        key-word arguments are passed to :py:meth:`geopandas.GeoDataFrame.to_file`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'geoms/{name}.geojson'
        to_wgs84: bool, optional
            Option to enforce writing GeoJSONs with WGS84(EPSG:4326) coordinates.
        \**kwargs:
            Additional keyword arguments that are passed to the
            `geopandas.to_file` function.
        """
        self.root._assert_write_mode()
        if len(self.geoms) == 0:
            self.logger.debug("No geoms data found, skip writing.")
            return
        for name, gdf in self.geoms.items():
            if not isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)) or len(gdf) == 0:
                self.logger.warning(
                    f"{name} object of type {type(gdf).__name__} not recognized"
                )
                continue
            self.logger.debug(f"Writing file {fn.format(name=name)}")
            _fn = join(self.root.path, fn.format(name=name))
            if not isdir(dirname(_fn)):
                os.makedirs(dirname(_fn))
            if to_wgs84 and (
                kwargs.get("driver") == "GeoJSON"
                or str(fn).lower().endswith(".geojson")
            ):
                gdf = gdf.to_crs(4326)
            gdf.to_file(_fn, **kwargs)
