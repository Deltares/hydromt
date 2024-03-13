# -*- coding: utf-8 -*-
"""Defines the CLI-API."""

import inspect
import logging
import typing
from typing import Dict, List, Union

from hydromt.data_catalog import DataCatalog

logger = logging.getLogger(__name__)


def get_model_components(model: str, component_types=None) -> Dict:
    """Get all model components, each described with the following keys.

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
    _EXPECTED_TYPES = [
        "int",
        "float",
        "str",
        "bool",
        "RasterDatasetSource",
        "GeoDatasetSource",
        "GeoDataframeSource",
    ]
    component_types = component_types or ["read", "write", "setup"]
    model_class = MODELS.load(model)
    members = inspect.getmembers(model_class)
    components = {}
    for name, member in members:
        if (
            name.startswith("_")
            or name.split("_")[0] not in component_types
            or not callable(member)
        ):
            continue
        signature = inspect.signature(member)
        components[name] = {
            "doc": inspect.getdoc(member),
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
            type = type if type in _EXPECTED_TYPES else _DEFAULT_TYPE
            if v.default == inspect._empty:  # required
                components[name]["required"].append((k, type))
            else:  # optional
                # TODO convert default value to string ?
                components[name]["optional"].append((k, type, v.default))

    return components


def get_datasets(data_libs: Union[List, str]) -> Dict:
    """Get all names of datasets sorted by data type.

        {
            "RasterDatasetSource": [],
            "GeoDatasetSource": [],
            "GeoDataframeSource": [],
        }

    Parameters
    ----------
    data_libs: (list of) str, Path, optional
        One or more paths to data catalog configuration files or names of predefined
        data catalogs. By default the data catalog is initiated without data entries.
        See :py:func:`~hydromt.data_adapter.DataCatalog.from_yml`
        for accepted yaml format.
    """
    data_catalog = DataCatalog(data_libs)
    dataset_sources = {
        "RasterDatasetSource": [],
        "GeoDatasetSource": [],
        "GeoDataframeSource": [],
    }
    for k, v in data_catalog.iter_sources():
        if v.data_type == "RasterDataset":
            dataset_sources["RasterDatasetSource"].append(k)
        elif v.data_type == "GeoDataFrame":
            dataset_sources["GeoDataframeSource"].append(k)
        elif v.data_type == "GeoDataset":
            dataset_sources["GeoDatasetSource"].append(k)
    return dataset_sources


def get_predifined_catalogs() -> Dict:
    """Get predefined catalogs.

    {
        <catalog_name> Dict: {}
    }

    """
    return DataCatalog().predefined_catalogs
