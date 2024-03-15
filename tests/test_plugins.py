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


# def test_plugins(mocker):
#     ep_lst = EntryPoints(
#         [
#             EntryPoint(
#                 name="test_model",
#                 value="hydromt.models.model_api:Model",
#                 group="hydromt.models",
#             )
#         ]
#     )
#     mocker.patch("hydromt.models.plugins._discover", return_value=ep_lst)
#     eps = plugins.get_plugin_eps()
#     assert "test_model" in eps
#     assert isinstance(eps["test_model"], EntryPoint)


# def test_plugin_duplicates(mocker):
#     ep_lst = plugins.get_general_eps().values()
#     mocker.patch("hydromt.models.plugins._discover", return_value=ep_lst)
#     eps = plugins.get_plugin_eps()
#     assert len(eps) == 0


# def test_load():
#     with pytest.raises(ValueError, match="Model plugin type not recognized"):
#         plugins.load(
#             EntryPoint(
#                 name="erfror",
#                 value="hydromt.data_catalog:DataCatalog",
#                 group="hydromt.data_catalog",
#             )
#         )
#     with pytest.raises(ImportError, match="Error while loading model plugin"):
#         plugins.load(
#             EntryPoint(
#                 name="error", value="hydromt.models:DataCatalog", group="hydromt.models"
#             )
#         )


# def test_global_models(mocker, has_xugrid):
#     _MODELS = ModelCatalog()
#     mocker.patch("hydromt._compat.HAS_XUGRID", has_xugrid)
#     keys = list(plugins.LOCAL_EPS.keys())
#     if not hydromt._compat.HAS_XUGRID:
#         keys.remove("mesh_model")
#     # set first local model as plugin for testing
#     _MODELS._plugins.append(keys[0])
#     assert isinstance(_MODELS[keys[0]], EntryPoint)
#     assert issubclass(_MODELS.load(keys[0]), Model)
#     assert keys[0] in _MODELS.__str__()
#     assert all([k in _MODELS for k in keys])  # eps
#     assert all([k in _MODELS.cls for k in keys])
#     with pytest.raises(ValueError, match="Unknown model"):
#         _MODELS["unknown"]
