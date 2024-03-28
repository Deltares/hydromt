"""Table component."""

import glob
import os
from os.path import basename, dirname, isdir, join, splitext
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries

from hydromt.components.base import ModelComponent
from hydromt.hydromt_step import hydromt_step
from hydromt.utils import DEFAULT_GEOM_FOLDER

if TYPE_CHECKING:
    from hydromt.models.model import Model


class GeomComponent(ModelComponent):
    """A component to manage geo-spatial geometries."""

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
        """
        if self._data is None:
            self._initialize_geoms()

        assert self._data is not None
        return self._data

    def _initialize_geoms(self, skip_read=False) -> None:
        """Initialize geoms."""
        if self._data is None:
            self._data = dict()
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
        assert self._data is not None
        if name in self._data:
            self._logger.warning(f"Replacing geom: {name}")

        if isinstance(geom, GeoSeries):
            geom = cast(GeoDataFrame, geom.to_frame())

        # Verify if a geom is set to model crs and if not sets geom to model crs
        model_crs = self._model.crs
        if model_crs and model_crs != geom.crs:
            geom.to_crs(model_crs.to_epsg(), inplace=True)
        self._data[name] = geom

    @hydromt_step
    def read(
        self, folder: str = DEFAULT_GEOM_FOLDER, extention: str = ".geojson", **kwargs
    ) -> None:
        r"""Read model geometries files at <root>/<filename>.

        key-word arguments are passed to :py:func:`geopandas.read_file`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root, may contain wildcards,
            by default ``geoms/\*.geojson``
        **kwargs:
            Additional keyword arguments that are passed to the
            `geopandas.read_file` function.
        """
        self._root._assert_read_mode()
        self._initialize_geoms(skip_read=True)
        fns = glob.glob(join(self._root.path, folder, "*"))
        for fn in fns:
            name = basename(fn).split(".")[0]
            ext = splitext(fn)[-1]
            if ext != extention:
                continue
            geom = cast(GeoDataFrame, gpd.read_file(fn, **kwargs))
            self._logger.debug(f"Reading model file {name} at {fn}.")

            self.set(geom=geom, name=name)

    @hydromt_step
    def write(
        self,
        folder: str = DEFAULT_GEOM_FOLDER,
        extention: str = ".geojson",
        paths: Optional[Dict[str, Path]] = None,
        to_wgs84: bool = False,
        **kwargs,
    ) -> None:
        r"""Write model geometries to a vector file (by default GeoJSON) at <root>/<fn>.

        key-word arguments are passed to :py:meth:`geopandas.GeoDataFrame.to_file`

        Parameters
        ----------
        fn : str, optional
            filename relative to model root and should contain a {name} placeholder,
            by default 'geoms/{name}.geojson'
        paths: Optional[Dict[str,Path]]
            a dictionary mapping names of components to file paths. Will override fn.
        to_wgs84: bool, optional
            Option to enforce writing GeoJSONs with WGS84(EPSG:4326) coordinates.
        \**kwargs:
            Additional keyword arguments that are passed to the
            `geopandas.to_file` function.
        """
        self._root._assert_write_mode()

        if len(self.data) == 0:
            self._logger.debug("No geoms data found, skip writing.")
            return

        for name, gdf in self.data.items():
            if len(gdf) == 0:
                self._logger.warning(f"{name} is empty. Skipping...")
                continue

            write_path = Path(
                join(
                    self._root.path,
                    folder.format(name=name),
                )
                + extention
            )
            self._logger.debug(f"Writing file {write_path}")

            write_folder = dirname(write_path)
            if not isdir(write_folder):
                os.makedirs(write_folder, exist_ok=True)

            if to_wgs84:
                gdf = gdf.to_crs(4326)
                assert gdf is not None

            gdf.to_file(write_path, **kwargs)
