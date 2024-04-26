"""Model Region class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple, cast

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from pyproj import CRS

from hydromt._typing.type_def import StrPath
from hydromt.components.base import ModelComponent
from hydromt.workflows.region import write_region

if TYPE_CHECKING:
    from hydromt.models import Model


class SpatialModelComponent(ModelComponent, ABC):
    """Define the model region."""

    DEFAULT_REGION_FILENAME = "region.geojson"

    def __init__(
        self,
        model: "Model",
        *,
        region_component: Optional[str] = None,
        filename: StrPath = DEFAULT_REGION_FILENAME,
    ) -> None:
        super().__init__(model)
        self._region_component = region_component
        self._region_filename = filename

    @property
    def bounds(self) -> Optional[np.ndarray]:
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
            "Property _data_region must be implemented in subclass."
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
        """
        self.root._assert_write_mode()
        if self._region_component is not None:
            self.logger.info(
                "Region is a reference to another component. Skipping writing..."
            )
            return

        write_region(
            self.region,
            filename=filename or self._region_filename,
            to_wgs84=to_wgs84,
            logger=self.logger,
            root=self.root,
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
            region_component = self.model.get_component(
                self._region_component,
                SpatialModelComponent,  # type: ignore # Only used for casting, not to create anything
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
