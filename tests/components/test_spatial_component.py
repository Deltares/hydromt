from typing import Optional, cast

import geopandas as gpd
from pytest_mock import MockerFixture

from hydromt.components.spatial import SpatialModelComponent
from hydromt.model.model import Model


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


def test_get_region_with_reference(world, mocker: MockerFixture):
    mocker.patch("hydromt.model.model.PLUGINS")

    class FakeModel(Model):
        def __init__(self):
            super().__init__(
                region_component="other",
                components={
                    "other": FakeSpatialComponent(self, gdf=world),
                    "component": FakeSpatialComponent(self, region_component="other"),
                },
            )

    model = FakeModel()
    assert model.region is world
    assert cast(SpatialModelComponent, model.component).region is world
