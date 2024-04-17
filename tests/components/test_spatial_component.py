import logging

import numpy as np
import pytest
from pytest_mock import MockerFixture

from hydromt.components.spatial import SpatialModelComponent
from hydromt.data_catalog import DataCatalog
from hydromt.models.model import Model


class FakeSpatialComponent(SpatialModelComponent):
    def write(self):
        self.write_region()

    def read(self):
        self.read_region()


def test_create_region_bbox(mocker: MockerFixture):
    data_catalog = mocker.Mock(spec_set=DataCatalog)
    model = mocker.Mock(
        spec=Model, data_catalog=data_catalog, logger=logging.getLogger()
    )
    component = FakeSpatialComponent(model)
    gdf = component.create_region(region={"bbox": [-1.0, -1.0, 1.0, 1.0]})
    assert gdf is component.region
    np.testing.assert_array_equal(gdf.total_bounds, component.bounds)
    assert gdf.crs == component.crs
    assert gdf.crs == 4326


def test_create_region_bbox_with_crs(mocker: MockerFixture):
    data_catalog = mocker.Mock(spec_set=DataCatalog)
    model = mocker.Mock(
        spec=Model, data_catalog=data_catalog, logger=logging.getLogger()
    )
    component = FakeSpatialComponent(model)
    gdf = component.create_region(region={"bbox": [-1.0, -1.0, 1.0, 1.0]}, crs=3035)
    assert gdf is component.region
    # Check that the data has been moved to the correct crs
    assert component.crs == 3035
    np.testing.assert_almost_equal(
        component.bounds, [2958198.4, -2404654.6, 3214455.5, -2178052.2], decimal=1
    )


def test_create_region_geom(mocker: MockerFixture, world):
    data_catalog = mocker.Mock(spec_set=DataCatalog)
    model = mocker.Mock(
        spec=Model, data_catalog=data_catalog, logger=logging.getLogger()
    )
    component = FakeSpatialComponent(model)
    component.create_region(region={"geom": world})
    assert component.region is world
    np.testing.assert_array_equal(world.total_bounds, component.bounds)


def test_create_region_geom_from_points_fails(geodf, mocker: MockerFixture):
    data_catalog = mocker.Mock(spec_set=DataCatalog)
    model = mocker.Mock(
        spec=Model, data_catalog=data_catalog, logger=logging.getLogger()
    )
    component = FakeSpatialComponent(model)
    with pytest.raises(ValueError, match=r"Region value.*"):
        component.create_region(region={"geom": geodf})


def test_create_referenced_component():
    model = Model(region_component="region")
    region_component = FakeSpatialComponent(model)
    reference_component = FakeSpatialComponent(model, region_component="region")
    model.add_component("region", region_component)
    model.add_component("region2", reference_component)
    region_component.create_region(region={"bbox": [-1.0, -1.0, 1.0, 1.0]})
    assert reference_component.region is region_component.region
