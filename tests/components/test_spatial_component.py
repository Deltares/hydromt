import logging

import numpy as np
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
