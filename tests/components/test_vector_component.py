from logging import DEBUG

import pytest
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


def test_write_empty_data(
    tmpdir, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    model = mocker.Mock(set=Model)
    model.root = mocker.Mock(set=ModelRoot)
    model.root.path = tmpdir
    vector = VectorComponent(model)
    with caplog.at_level(DEBUG):
        vector.write()
    assert "No vector data found, skip writing." in caplog.text
