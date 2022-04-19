"""This file describes an API for HydroMT, used by e.g. HydroMT-Dash to dynamically generate the inputs.
"""
import typing
import inspect

from hydromt.models import ENTRYPOINTS


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
    for name, member in members:
        if name.startswith("_") or name in _SKIP_METHODS or not callable(member):
            continue
        signature = inspect.signature(member)
        components[name] = {
            # 'doc': member.__doc__,
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


#%% test
# get_model_components('wflow')
