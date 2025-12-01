"""Geoms component."""

import logging
from glob import glob
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.testing import assert_geodataframe_equal

from hydromt._utils.naming_convention import _expand_uri_placeholders
from hydromt.model.components.base import ModelComponent
from hydromt.model.components.spatial import SpatialModelComponent
from hydromt.model.steps import hydromt_step
from hydromt.typing.crs import CRS

if TYPE_CHECKING:
    from hydromt.model.model import Model

logger = logging.getLogger(__name__)


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
    def crs(self) -> Optional[CRS]:
        """
        Return the coordinate reference system associated with the geometries.

        If multiple GeoDataFrames are present with differing CRS values, the CRS of the first
        GeoDataFrame that defines one is returned. If none of the stored geometries provide a
        CRS, the method returns None.

        Returns
        -------
        CRS or None
            The CRS of the first geometry with a defined coordinate reference system, or None
            if no CRS is available.
        """
        for geom in self.data.values():
            if geom.crs is not None:
                return geom.crs
        return None

    @property
    def _region_data(self) -> Optional[GeoDataFrame]:
        """
        Return the region as the geometric union of all polygonal features stored in the data dictionary.

        Each GeoDataFrame in ``self.data`` is exploded to ensure that multi-part geometries are treated
        as individual components before computing the union. Only geometries with area contribute to the
        resulting region; non-area features such as points and lines are ignored.

        Returns
        -------
        geopandas.GeoDataFrame or None
            A GeoDataFrame containing the unified region geometry, or None if no polygonal features
            are available.
        """
        # Use the union of all geometries as region
        if len(self.data) == 0:
            return None

        # Flatten all geometries from all GeoDataFrames
        all_geoms = [
            geom
            for gdf in self.data.values()
            if not gdf.empty
            for geom in gdf.geometry.explode(index_parts=False)
            if geom is not None and not geom.is_empty and geom.geom_type == "Polygon"
        ]
        union_geom = gpd.GeoSeries(all_geoms, crs=self.crs).union_all()
        return gpd.GeoDataFrame(geometry=[union_geom], crs=self.crs)

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
            logger.warning(f"Replacing geom: {name}")

        if isinstance(geom, GeoSeries):
            geom = cast(GeoDataFrame, geom.to_frame())

        # Verify if a geom is set to model crs and if not sets geom to model crs
        model_crs = self.model.crs or self.crs
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
        path_glob, _, regex = _expand_uri_placeholders(str(read_path))
        paths = glob(path_glob)
        for p in paths:
            name = ".".join(regex.match(p).groups())  # type: ignore
            geom = cast(GeoDataFrame, gpd.read_file(p, **kwargs))
            logger.debug(f"Reading model file {name} at {p}.")

            self.set(geom=geom, name=name)

    @hydromt_step
    def write(
        self,
        filename: Optional[str] = None,
        *,
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
            Can be a relative path.
        to_wgs84: bool, optional
            If True, the geoms will be reprojected to WGS84(EPSG:4326) before they are written.
        **kwargs:
            Additional keyword arguments that are passed to the
            `geopandas.to_file` function.
        """
        self.root._assert_write_mode()

        if len(self.data) == 0:
            logger.info(
                f"{self.model.name}.{self.name_in_model}: No geoms data found, skip writing."
            )
            return

        filename = filename or self._filename

        for name, gdf in self.data.items():
            if len(gdf) == 0:
                logger.warning(
                    f"{self.model.name}.{self.name_in_model}: {name} is empty. Skipping..."
                )
                continue

            write_path = self.root.path / filename.format(name=name)
            write_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"{self.model.name}.{self.name_in_model}: Writing geoms to {write_path}."
            )

            if to_wgs84 and (
                kwargs.get("driver") == "GeoJSON"
                or write_path.suffix.lower() == ".geojson"
            ):
                gdf.to_crs(epsg=4326, inplace=True)

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
