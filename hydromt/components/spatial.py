"""Model Region class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple, cast

import geopandas as gpd
from geopandas import GeoDataFrame
from pyproj import CRS

from hydromt._typing.type_def import StrPath
from hydromt.components.base import ModelComponent
from hydromt.io.writers import write_region

if TYPE_CHECKING:
    from hydromt.model import Model


class SpatialModelComponent(ModelComponent, ABC):
    """Base spatial model component for GIS components."""

    def __init__(
        self,
        model: "Model",
        *,
        region_component: Optional[str] = None,
        region_filename: str = "region.geojson",
    ) -> None:
        """
        Initialize a SpatialModelComponent.

        This component serves as a base class for components that are geospatial and
        require a region.

        To re-use in your won component, make sure you implement the `_region_data`
        property.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        region_component: str, optional
            The name of the region component to use as reference for this component's
            region in case the region of this new component depends on the region of a
            different component. If None, the region will be set based on the
            `_region_data` property of the component itself.
        region_filename: str
            The path to use for writing the region data to a file.
            By default "region.geojson".
        """
        super().__init__(model)
        self._region_component = region_component
        self._region_filename = region_filename

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Return the total bounds of the model region."""
        return self.region.total_bounds if self.region is not None else None

    @property
    def region(self) -> Optional[GeoDataFrame]:
        """Provide access to the underlying GeoDataFrame data of the model region."""
        region_from_reference = self._get_region_from_reference()
        return (
            region_from_reference
            if region_from_reference is not None
            else self._region_data
        )

    @property
    @abstractmethod
    def _region_data(self) -> Optional[GeoDataFrame]:
        """Implement this property in order to provide the region.

        This function will be called by the `region` property if no reference component is set.
        """
        raise NotImplementedError(
            "Property _region_data must be implemented in subclass."
        )

    @property
    def crs(self) -> Optional[CRS]:
        """Provide access to the CRS of the model region."""
        return self.region.crs if self.region is not None else None

    def write_region(
        self,
        *,
        filename: Optional[StrPath] = None,
        to_wgs84=False,
        **write_kwargs,
    ) -> None:
        """Write the model region to file.

        This function should be called from within the `write` function of the component inheriting from this class.

        Parameters
        ----------
        filename : str, optional
            The filename to write the region to. If None, the filename provided at initialization is used.
        to_wgs84 : bool, optional
            If True, the region is reprojected to WGS84 before writing.
        **write_kwargs:
            Additional keyword arguments passed to the `geopandas.GeoDataFrame.to_file` function.
        """
        self.root._assert_write_mode()
        if self._region_component is not None:
            self.logger.info(
                "Region is a reference to another component. Skipping writing..."
            )
            return

        if self.region is None:
            self.logger.info("No region data available to write.")
            return

        write_region(
            self.region,
            filename=filename or self._region_filename,
            to_wgs84=to_wgs84,
            logger=self.logger,
            root_path=self.root.path,
            **write_kwargs,
        )

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
            region_component = cast(
                SpatialModelComponent, self.model.get_component(self._region_component)
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
