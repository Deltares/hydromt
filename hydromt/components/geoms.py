"""Table component."""

import os
from glob import glob
from os.path import dirname, isdir, join
from pathlib import Path
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
from hydromt.metadata_resolver import ConventionResolver

if TYPE_CHECKING:
    from hydromt.models.model import Model

_DEFAULT_GEOMS_FILENAME = "geoms/{name}.geojson"


class GeomsComponent(ModelComponent):
    """A component to manage geo-spatial geometries."""

    def __init__(self, model: "Model", filename: str = _DEFAULT_GEOMS_FILENAME):
        """Initialize a GeomComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            The path to use for reading and writing of component data by default.
            by default DEFAULT_GEOMS_FILENAME
        """
        self._data: Optional[Dict[str, Union[GeoDataFrame, GeoSeries]]] = None
        self._filename = filename
        super().__init__(model=model)

    @property
    def data(self) -> Dict[str, Union[GeoDataFrame, GeoSeries]]:
        """Model geometries.

        Return dict of geopandas.GeoDataFrame or geopandas.GeoDataSeries
        """
        if self._data is None:
            self._initialize()

        assert self._data is not None
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize geoms."""
        if self._data is None:
            self._data = dict()
            if self._root.is_reading_mode() and not skip_read:
                self.read()

    def set(self, geom: Union[GeoDataFrame, GeoSeries], name: str):
        """Add data to the geom component.

        Arguments
        ---------
        geom: geopandas.GeoDataFrame or geopandas.GeoSeries
            New geometry data to add
        name: str
            Geometry name.
        """
        self._initialize()
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
    def read(self, filename: Optional[str] = None, **kwargs) -> None:
        r"""Read model geometries files at <root>/<filename>.

        key-word arguments are passed to :py:func:`geopandas.read_file`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root. should contain a {name} placeholder
            which will be used to determine the names/keys of the geometries.
            if None, the path that was provided at init will be used.
        **kwargs:
            Additional keyword arguments that are passed to the
            `geopandas.read_file` function.
        """
        self._root._assert_read_mode()
        self._initialize(skip_read=True)
        read_path = join(self._root.path, filename or self._filename)
        fn_glob, _, regex = ConventionResolver()._expand_uri_placeholders(read_path)
        fns = glob(fn_glob)
        for fn in fns:
            name = ".".join(regex.match(fn).groups())  # type: ignore
            geom = cast(GeoDataFrame, gpd.read_file(fn, **kwargs))
            self._logger.debug(f"Reading model file {name} at {fn}.")

            self.set(geom=geom, name=name)

    @hydromt_step
    def write(
        self,
        filename: Optional[str] = None,
        to_wgs84: bool = False,
        **kwargs,
    ) -> None:
        r"""Write model geometries to a vector file (by default GeoJSON) at <root>/<filename>.

        key-word arguments are passed to :py:meth:`geopandas.GeoDataFrame.to_file`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root. should contain a {name} placeholder
            which will be used to determine the names/keys of the geometries.
            if None, the path that was provided at init will be used.
        to_wgs84: bool, optional
            If True, the geoms will be reprojected to WGS84(EPSG:4326) before they are written.
        **kwargs:
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

            geom_filename = filename or self._filename

            write_path = Path(
                join(
                    self._root.path,
                    geom_filename.format(name=name),
                )
            )

            self._logger.debug(f"Writing file {write_path}")

            write_folder = dirname(write_path)
            if not isdir(write_folder):
                os.makedirs(write_folder, exist_ok=True)

            if to_wgs84:
                # no idea why pyright complains about the next line
                # so just ignoring it
                gdf.to_crs(epsg=4326, inplace=True)  # type: ignore

            gdf.to_file(write_path, **kwargs)
