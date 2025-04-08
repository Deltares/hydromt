from typing import List, Type

import pytest
from importlib_metadata import EntryPoint, EntryPoints, entry_points
from pytest_mock import MockerFixture

from hydromt.data_catalog.drivers import __hydromt_eps__ as driver_eps
from hydromt.data_catalog.predefined_catalog import __hydromt_eps__ as catalog_eps
from hydromt.model.components import __hydromt_eps__ as component_eps
from hydromt.model.components.grid import GridComponent
from hydromt.model.model import Model


def test_core_component_plugins(PLUGINS):
    components = PLUGINS.component_plugins
    obj_names = [obj.__name__ for obj in components.values()]
    assert all([d in obj_names for d in component_eps])


def test_core_model_plugins(PLUGINS):
    models = PLUGINS.model_plugins
    assert models == {"model": Model}


def test_core_driver_plugins(PLUGINS):
    drivers = PLUGINS.driver_plugins
    obj_names = [obj.__name__ for obj in drivers.values()]
    assert all([d in obj_names for d in driver_eps])


def test_core_catalog_plugins(PLUGINS):
    catalogs = PLUGINS.catalog_plugins
    obj_names = [obj.__name__ for obj in catalogs.values()]
    assert all([d in obj_names for d in catalog_eps])


def test_summary(PLUGINS):
    component_summary = PLUGINS.component_summary()
    model_summary = PLUGINS.model_summary()
    driver_summary = PLUGINS.driver_summary()

    assert component_summary.startswith("Component plugins:")
    assert model_summary.startswith("Model plugins:")
    assert driver_summary.startswith("Driver plugins:")
    # check for some specific plugins
    assert "GridComponent" in component_summary
    assert "model" in model_summary
    assert "pyogrio" in driver_summary


def _patch_plugin_entry_point(
    mocker: MockerFixture, component_names: List[str], component_class: Type
):
    # create a mocked version of hydromt.plugins.entry_points
    mock_module = mocker.MagicMock(__hydromt_eps__=component_names)
    if component_class:
        for c in component_names:
            mock_module.__setattr__(c, component_class)

    mock_single_entrypoint = mocker.create_autospec(EntryPoint, spec_set=True)
    mock_single_entrypoint.dist.version = "999.9.9"
    mock_single_entrypoint.dist.name = "hydromt-test"

    mock_multiple_entrypoints = mocker.create_autospec(EntryPoints, spec_set=True)
    mock_multiple_entrypoints.__iter__.return_value = [mock_single_entrypoint]
    mock_single_entrypoint.load.return_value = mock_module

    func = mocker.create_autospec(entry_points, spec_set=True)
    func.return_value = mock_multiple_entrypoints

    return func


def test_errors_on_duplicate_plugins(PLUGINS, mocker):
    mocked_entrypoints = _patch_plugin_entry_point(
        mocker, ["GridComponent", "GridComponent"], GridComponent
    )
    mocker.patch("hydromt.plugins.entry_points", new=mocked_entrypoints)
    with pytest.raises(ValueError, match="Conflicting definitions for "):
        _ = PLUGINS.component_plugins


def test_errors_on_non_existing_plugins(PLUGINS, mocker):
    mocked_entrypoints = _patch_plugin_entry_point(mocker, ["SomeComponent"], None)
    mocker.patch("hydromt.plugins.entry_points", new=mocked_entrypoints)
    with pytest.raises(ValueError, match="is not a valid "):
        _ = PLUGINS.component_plugins


def test_errors_on_wrong_plugin_type(PLUGINS, mocker):
    mocked_entrypoints = _patch_plugin_entry_point(mocker, ["SomeComponent"], Model)
    mocker.patch("hydromt.plugins.entry_points", new=mocked_entrypoints)
    with pytest.raises(ValueError, match="is not a valid "):
        _ = PLUGINS.component_plugins
