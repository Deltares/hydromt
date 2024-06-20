from logging import Logger

import xarray as xr
from pytest_mock import MockerFixture

from hydromt.model.components.vector import VectorComponent
from hydromt.model.model import Model
from hydromt.model.root import ModelRoot


def test_empty_data(tmpdir, mocker: MockerFixture):
    model = mocker.Mock(set=Model)
    model.root = mocker.Mock(set=ModelRoot)
    model.root.path = tmpdir
    vector = VectorComponent(model)
    xr.testing.assert_identical(vector.data, xr.Dataset())


def test_write_empty_data(tmpdir, mocker: MockerFixture):
    model = mocker.Mock(set=Model)
    model.root = mocker.Mock(set=ModelRoot)
    model.root.path = tmpdir
    model.logger = mocker.Mock(spec_set=Logger)
    vector = VectorComponent(model)
    vector.write()
    model.logger.debug.assert_called_once_with("No vector data found, skip writing.")
