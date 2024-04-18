from typing import Optional

import geopandas as gpd

from hydromt.components.spatial import SpatialModelComponent
from hydromt.models.model import Model


class FakeSpatialComponent(SpatialModelComponent):
    def __init__(
        self,
        model: Model,
        *,
        gdf: Optional[gpd.GeoDataFrame] = None,
        region_component: Optional[str] = None,
    ):
        super().__init__(model, region_component=region_component)
        self._gdf = gdf

    def write(self):
        self.write_region()

    def read(self):
        pass

    @property
    def _region_data(self):
        return self._gdf


def test_get_region_with_reference(world):
    model = Model()
    component = FakeSpatialComponent(model, region_component="other")
    referenced = FakeSpatialComponent(model, gdf=world)
    model.add_component("other", referenced)
    model.add_component("region", component)

    assert component.region is world
