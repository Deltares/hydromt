from typing import List

import pytest
from importlib_metadata import EntryPoint, EntryPoints, entry_points
from pytest_mock import MockerFixture

from hydromt.components.base import ModelComponent
from hydromt.components.config import ConfigComponent
from hydromt.components.geoms import GeomsComponent
from hydromt.components.grid import GridComponent
from hydromt.components.region import ModelRegionComponent
from hydromt.components.tables import TablesComponent
from hydromt.components.vector import VectorComponent
from hydromt.models.model import Model
from hydromt.plugins import PLUGINS


def test_core_component_plugins():
    components = PLUGINS.component_plugins
    assert components == {
        "ConfigComponent": ConfigComponent,
        "GeomsComponent": GeomsComponent,
        "GridComponent": GridComponent,
        "ModelComponent": ModelComponent,
        "VectorComponent": VectorComponent,
        "TablesComponent": TablesComponent,
        "ModelRegionComponent": ModelRegionComponent,
    }


def test_core_model_plugins():
    models = PLUGINS.model_plugins
    assert models == {"Model": Model}


def test_summary():
    component_summary = PLUGINS.component_summary()
    model_summary = PLUGINS.model_summary()
    driver_summary = PLUGINS.driver_summary()
    assert "Component plugins:" in component_summary
    assert "Model plugins:" in model_summary
    assert "ModelRegionComponent" in component_summary
    assert "GridComponent" in component_summary
    assert "ModelComponent" in component_summary
    assert "Model" in model_summary
    assert "Driver Plugins:" in driver_summary
    assert "PyogrioDriver" in driver_summary
    assert "RasterDatasetDriver" not in driver_summary  # No ABCs in Plugins
    assert "harmonize_dims" not in driver_summary  # only drivers in Plugins


def _patch_plugin_entry_point(mocker: MockerFixture, component_names: List[str]):
    ### SETUP
    PLUGINS._component_plugins = None
    PLUGINS._model_plugins = None

    mock_single_entrypoint = mocker.create_autospec(EntryPoint, spec_set=True)
    mock_single_entrypoint.dist.version = "999.9.9"
    mock_single_entrypoint.dist.name = "hydromt-test"
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
    PLUGINS._driver_plugins = None
    PLUGINS._model_plugins = None
    yield
    PLUGINS._component_plugins = None
    PLUGINS._driver_plugins = None
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
