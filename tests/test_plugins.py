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
