from typing import List

import pytest
from importlib_metadata import EntryPoint, EntryPoints, entry_points
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
    ### SETUP
    PLUGINS._component_plugins = None
    PLUGINS._model_plugins = None

    mock_single_entrypoint = mocker.create_autospec(EntryPoint, spec_set=True)
    mock_multiple_entrypoints = mocker.create_autospec(EntryPoints, spec_set=True)
    mock_multiple_entrypoints.__iter__.return_value = [mock_single_entrypoint]

    mock_module = mocker.MagicMock(__all__=component_names)
    mock_single_entrypoint.load.return_value = mock_module
    mocked_components = []
    for c in component_names:
        mock_component = mocker.create_autospec(
            ModelComponent, spec_set=True, instance=True
        )
        mock_module.__setattr__(c, mock_component)
        mocked_components.append(mock_component)

    func = mocker.create_autospec(entry_points, spec_set=True)
    func.return_value = mock_multiple_entrypoints

    return func, mocked_components


@pytest.fixture()
def _reset_plugins():
    PLUGINS._component_plugins = None
    PLUGINS._model_plugins = None
    yield
    PLUGINS._component_plugins = None
    PLUGINS._model_plugins = None


@pytest.mark.usefixtures("_reset_plugins")
def test_errors_on_duplicate_plugins(
    mocker,
):
    mocked_entrypoints, _ = _patch_plugin_entry_point(
        mocker, ["TestModelComponent", "TestModelComponent"]
    )

    mocker.patch("hydromt.plugins.entry_points", new=mocked_entrypoints)
    with pytest.raises(ValueError, match="Conflicting definitions for "):
        _ = PLUGINS.component_plugins


@pytest.mark.usefixtures("_reset_plugins")
def test_discover_mock_plugin(mocker):
    PLUGINS._component_plugins = None
    mock_entrypoints, mocked_components = _patch_plugin_entry_point(
        mocker, ["TestModelComponent", "OtherTestModelComponent"]
    )
    mocker.patch("hydromt.plugins.entry_points", new=mock_entrypoints)
    components = PLUGINS.component_plugins
    assert components == {
        "TestModelComponent": mocked_components[0],
        "OtherTestModelComponent": mocked_components[1],
    }