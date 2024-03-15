from typing import List
from unittest.mock import MagicMock, create_autospec

import pytest
from pytest_mock import MockerFixture

from hydromt.components.base import ModelComponent
from hydromt.components.grid import GridComponent
from hydromt.components.region import ModelRegionComponent
from hydromt.components.root import ModelRootComponent
from hydromt.models.model import Model
from hydromt.plugins import PLUGINS


def test_core_component_plugins():
    components = PLUGINS.component_plugins
    assert components == {
        "ModelRegionComponent": ModelRegionComponent,
        "GridComponent": GridComponent,
        "ModelComponent": ModelComponent,
        "ModelRootComponent": ModelRootComponent,
    }


def test_core_model_plugins():
    models = PLUGINS.model_plugins
    assert models == {"Model": Model}


def test_summary():
    summary = PLUGINS.summary()
    assert "component plugins:" in summary
    assert "model plugins:" in summary
    assert "ModelRegionComponent" in summary
    assert "GridComponent" in summary
    assert "ModelComponent" in summary
    assert "ModelRootComponent" in summary
    assert "Model" in summary


def _patch_plugin_entry_point(mocker: MockerFixture, component_names: List[str]):
    mock_single_entrypoint = MagicMock()
    mock_multiple_entrypoints = MagicMock()
    mock_multiple_entrypoints.__iter__.return_value = [mock_single_entrypoint]

    mock_module = MagicMock(__all__=component_names)
    mock_single_entrypoint.load.return_value = mock_module
    mocked_components = []
    for c in component_names:
        mock_component = create_autospec(ModelComponent, spec_set=True, instance=True)
        mock_module.__setattr__(c, mock_component)
        mocked_components.append(mock_component)

    func = MagicMock()
    func.return_value = mock_multiple_entrypoints

    return func, mocked_components


def test_errors_on_duplicate_plugins(mocker):
    mocked_entrypoints, _ = _patch_plugin_entry_point(
        mocker, ["TestModelComponent", "TestModelComponent"]
    )

    mocker.patch("hydromt.plugins.entry_points", new=mocked_entrypoints)
    with pytest.raises(ValueError, match="Conflicting definitions for "):
        _ = PLUGINS.component_plugins


def test_discover_mock_plugin(mocker):
    mock_entrypoints, mocked_components = _patch_plugin_entry_point(
        mocker, ["TestModelComponent", "OtherTestModelComponent"]
    )
    with mocker.patch("hydromt.plugins.entry_points", new=mock_entrypoints):
        components = PLUGINS.component_plugins
    assert components == {
        "TestModelComponent": mocked_components[0],
        "OtherTestModelComponent": mocked_components[1],
    }
