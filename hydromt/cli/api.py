# -*- coding: utf-8 -*-
"""This file describes an API for HydroMT, used by e.g. HydroMT-Dash to dynamically generate the inputs.
"""
from typing import List, Dict, Union
import typing
import inspect

from ..models import ENTRYPOINTS
from ..data_catalog import DataCatalog


def get_model_components(
    model: str, component_types=["read", "write", "setup"]
) -> Dict:
    """Get all model components, each described with the following keys

        {
            <component_name> dict: {
                "doc" str: doc string
                "required" List: tuples of argument name and dtype,
                "optional" List: tuples of argument name, dtype and default value,
                "kwargs" bool: whether the component accepts key-word arguments,
            }
        }

    Parameters
    ----------
    model : str
        model name
    component_types: list
        model components types to return, by default ['read','write','setup']
    """
    _DEFAULT_TYPE = "str"
    _EXCEPTED_TYPES = [
        "int",
        "float",
        "str",
        "bool",
        "RasterDatasetSource",
        "GeoDatasetSource",
        "GeoDataframeSource",
    ]
    model_class = ENTRYPOINTS[model].load()
    members = inspect.getmembers(model_class)
    components = {}
    docs = []
    for name, member in members:
        if (
            name.startswith("_")
            or not name.split("_")[0] in component_types
            or not callable(member)
        ):
            continue
        signature = inspect.signature(member)
        components[name] = {
            "doc": member.__doc__,
            "required": [],
            "optional": [],
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
                # TODO convert default value to string ?
                components[name]["optional"].append((k, type, v.default))

    return components


def get_datasets(data_libs: Union[List, str]) -> Dict:
    """Get all names of datasets sorted by data type

        {
            "RasterDatasetSource": [],
            "GeoDatasetSource": [],
            "GeoDataframeSource": [],
        }

    Parameters
    ----------
    data_libs: (list of) str, Path, optional
        One or more paths to data catalog yaml files or names of predefined data catalogs.
        By default the data catalog is initiated without data entries.
        See :py:func:`~hydromt.data_adapter.DataCatalog.from_yml` for accepted yaml format.
    """
    data_catalog = DataCatalog(data_libs)
    datasets = data_catalog.sources
    dataset_sources = {
        "RasterDatasetSource": [],
        "GeoDatasetSource": [],
        "GeoDataframeSource": [],
    }
    for k, v in datasets.items():
        if v.data_type == "RasterDataset":
            dataset_sources["RasterDatasetSource"].append(k)
        elif v.data_type == "GeoDataFrame":
            dataset_sources["GeoDataframeSource"].append(k)
        elif v.data_type == "GeoDataset":
            dataset_sources["GeoDatasetSource"].append(k)
    return dataset_sources


def get_predifined_catalogs() -> Dict:
    """Get predefined catalogs

    {
        <catalog_name> Dict: {}
    }

    """
    return DataCatalog().predefined_catalogs
