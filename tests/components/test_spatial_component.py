from typing import Optional, cast

import geopandas as gpd
from pytest_mock import MockerFixture

from hydromt.model.components.spatial import SpatialModelComponent
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


def test_spatialmodelcomponent_test_equal_identical(mock_model, world):
    comp1 = FakeSpatialComponent(mock_model, gdf=world)
    comp2 = FakeSpatialComponent(mock_model, gdf=world.copy())
    eq, errors = comp1.test_equal(comp2)
    assert eq
    assert errors == {}


def test_spatialmodelcomponent_test_equal_class_mismatch(mock_model, world):
    comp = FakeSpatialComponent(mock_model, gdf=world)

    class Dummy:
        pass

    dummy = Dummy()
    eq, errors = comp.test_equal(dummy)
    assert not eq
    assert "__class__" in errors


def test_spatialmodelcomponent_test_equal_missing_region(mock_model, world):
    comp1 = FakeSpatialComponent(mock_model, gdf=world)
    comp2 = FakeSpatialComponent(mock_model, gdf=None)
    eq, errors = comp1.test_equal(comp2)
    assert not eq
    assert "data" in errors
    assert "missing" in errors["data"]


def test_spatialmodelcomponent_test_equal_region_not_equal(mock_model, world):
    comp1 = FakeSpatialComponent(mock_model, gdf=world)
    # Create a different region by dropping a row
    comp2 = FakeSpatialComponent(mock_model, gdf=world.iloc[:-1])
    eq, errors = comp1.test_equal(comp2)
    assert not eq
    assert any("Not equal" in v for v in errors.values()), errors
