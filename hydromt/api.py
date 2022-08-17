"""This file describes an API for HydroMT, used by e.g. HydroMT-Dash to dynamically generate the inputs.
"""
import typing
import inspect
import json

from hydromt.models import ENTRYPOINTS
from hydromt import DataCatalog


def get_model_components(model: str):
    """Return model components with their arguments.

    Parameters
    ----------
    model : str
        model name

    Returns
    -------
    components: dict
        dict of model components with their arguments
    """
    _DEFAULT_TYPE = "str"
    _EXCEPTED_TYPES = ["int", "float", "str", "bool"]
    _SKIP_METHODS = ["build", "update", "clip"]
    model_class = ENTRYPOINTS[model].load()
    members = inspect.getmembers(model_class)
    components = {}
    docs = []
    for name, member in members:
        if name.startswith("_") or name in _SKIP_METHODS or not callable(member):
            continue
        signature = inspect.signature(member)
        components[name] = {
            'doc': member.__doc__,
            "required": [],
            "optional": [],
            "args": False,
            "kwargs": False,
        }
        for k, v in signature.parameters.items():
            if k in ["self"]:
                continue
            elif k in ["args", "kwargs"]:
                components[name][k] = True
                continue
            annotation = v.annotation
            if typing.get_origin(annotation) == typing.Union:
                annotation = typing.get_args(v.annotation)[0]
            type = getattr(annotation, "__name__", _DEFAULT_TYPE)
            type = type if type in _EXCEPTED_TYPES else _DEFAULT_TYPE
            if v.default == inspect._empty:  # required
                components[name]["required"].append((k, type))
            else:  # optional
                components[name]["optional"].append((k, type, v.default))
                
    return components

# def get_data_catalog(data_catalog_name):
#     data_catalog = DataCatalog()
#     print(data_catalog.predefined_catalogs)

# get_data_catalog("deltares_data")
# print(DataCatalog.predefined_catalogs)

# model_plugins = ['wflow', 'fiat', 'delwaq', 'sfincs']

# for plugin in model_plugins:
#     components = get_model_components(plugin)
#     with open(f'{plugin}_api.json', 'w') as f:
#         json.dump(components, f, indent=2)