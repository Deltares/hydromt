from logging import DEBUG
from pathlib import Path

import pytest
import xarray as xr
from pytest_mock import MockerFixture

from hydromt.model.components.vector import VectorComponent
from hydromt.model.model import Model
from hydromt.model.root import ModelRoot


def test_empty_data(tmp_path: Path, mocker: MockerFixture):
    model = mocker.Mock(set=Model)
    model.root = mocker.Mock(set=ModelRoot)
    model.root.path = tmp_path
    vector = VectorComponent(model)
    xr.testing.assert_identical(vector.data, xr.Dataset())


def test_write_empty_data(
    tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    model = mocker.Mock(set=Model)
    model.root = mocker.Mock(set=ModelRoot)
    model.root.path = tmp_path
    model.name = "foo"
    vector = VectorComponent(model)
    model.components = {}
    model.components["vector"] = vector
    with caplog.at_level(DEBUG):
        vector.write()
    assert "foo.vector: No vector data found, skip writing." in caplog.text
