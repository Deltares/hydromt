"""Geoms component."""

import os
from glob import glob
from os.path import dirname, isdir, join
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Optional,
    Tuple,
    Union,
    cast,
)

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import box

from hydromt.hydromt_step import hydromt_step
from hydromt.metadata_resolver import ConventionResolver
from hydromt.model.components.base import ModelComponent
from hydromt.model.components.spatial import SpatialModelComponent

if TYPE_CHECKING:
    from hydromt.model.model import Model


class GeomsComponent(SpatialModelComponent):
    """A component to manage geo-spatial geometries.

    It contains a dictionary of geopandas GeoDataFrames.
    """

    def __init__(
        self,
        model: "Model",
        *,
        filename: str = "geoms/{name}.geojson",
        region_component: Optional[str] = None,
        region_filename: str = "geoms/geoms_region.geojson",
    ):
        """Initialize a GeomsComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            The path to use for reading and writing of component data by default.
            by default "geoms/{name}.geojson" ie one file per geodataframe in the data dictionary.
        region_component: str, optional
            The name of the region component to use as reference for this component's
            region. If None, the region will be set to the union of all geometries in
            the data dictionary.
        region_filename: str
            The path to use for writing the region data to a file. By default
            "geoms/geoms_region.geojson".
        """
        self._data: Optional[Dict[str, Union[GeoDataFrame, GeoSeries]]] = None
        self._filename: str = filename
        super().__init__(
            model=model,
            region_component=region_component,
            region_filename=region_filename,
        )

    @property
    def data(self) -> Dict[str, Union[GeoDataFrame, GeoSeries]]:
        """Model geometries.

        Return dict of geopandas.GeoDataFrame or geopandas.GeoSeries
        """
        if self._data is None:
            self._initialize()

        assert self._data is not None
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize geoms."""
        if self._data is None:
            self._data = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    @property
    def _region_data(self) -> Optional[GeoDataFrame]:
        # Use the total bounds of all geometries as region
        if len(self.data) == 0:
            return None
        bounds = np.column_stack([geom.bounds for geom in self.data.values()])
        total_bounds = (
            bounds[0].min(),
            bounds[1].min(),
            bounds[2].max(),
            bounds[3].max(),
        )
        region = gpd.GeoDataFrame(
            geometry=[gpd.GeoSeries(box(total_bounds))], crs=self.model.crs
        )

        return region

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
            self.logger.warning(f"Replacing geom: {name}")

        if isinstance(geom, GeoSeries):
            geom = cast(GeoDataFrame, geom.to_frame())

        # Verify if a geom is set to model crs and if not sets geom to model crs
        model_crs = self.model.crs
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
        self.root._assert_read_mode()
        self._initialize(skip_read=True)
        f = filename or self._filename
        read_path = self.root.path / f
        fn_glob, _, regex = ConventionResolver()._expand_uri_placeholders(
            str(read_path)
        )
        fns = glob(fn_glob)
        for fn in fns:
            name = ".".join(regex.match(fn).groups())  # type: ignore
            geom = cast(GeoDataFrame, gpd.read_file(fn, **kwargs))
            self.logger.debug(f"Reading model file {name} at {fn}.")

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
        self.root._assert_write_mode()

        if len(self.data) == 0:
            self.logger.debug("No geoms data found, skip writing.")
            return

        for name, gdf in self.data.items():
            if len(gdf) == 0:
                self.logger.warning(f"{name} is empty. Skipping...")
                continue

            geom_filename = filename or self._filename

            write_path = Path(
                join(
                    self.root.path,
                    geom_filename.format(name=name),
                )
            )

            self.logger.debug(f"Writing file {write_path}")

            write_folder = dirname(write_path)
            if not isdir(write_folder):
                os.makedirs(write_folder, exist_ok=True)

            if to_wgs84 and (
                kwargs.get("driver") == "GeoJSON"
                or str(write_path).lower().endswith(".geojson")
            ):
                # no idea why pyright complains about the next line
                # so just ignoring it
                gdf.to_crs(epsg=4326, inplace=True)  # type: ignore

            gdf.to_file(write_path, **kwargs)

    def test_equal(self, other: ModelComponent) -> Tuple[bool, Dict[str, str]]:
        """Test if two GeomsComponents are equal.

        Parameters
        ----------
        other: GeomsComponent
            The other GeomsComponent to compare with.

        Returns
        -------
        tuple[bool, dict[str, str]]
            True if the components are equal, and a dict with the associated errors per property checked.
        """
        eq, errors = super().test_equal(other)
        if not eq:
            return eq, errors
        other_geoms = cast(GeomsComponent, other)
        for name, gdf in self.data.items():
            if name not in other_geoms.data:
                errors[name] = "Geom not found in other component."
            try:
                assert_geodataframe_equal(
                    gdf,
                    other_geoms.data[name],
                    check_like=True,
                    check_less_precise=True,
                )
            except AssertionError as e:
                errors[name] = str(e)

        return len(errors) == 0, errors
