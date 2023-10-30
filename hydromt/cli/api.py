# -*- coding: utf-8 -*-
"""Defines the CLI-API."""
import inspect
import logging
import typing
from typing import Dict, List, Optional, Union

from hydromt.gis_utils import utm_crs

from .. import workflows
from ..data_catalog import DataCatalog
from ..models import MODELS

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


def get_region(
    region: dict,
    data_libs: Optional[Union[List, str]] = None,
    hydrography_fn: str = "merit_hydro",
    basin_index_fn: str = "merit_hydro_index",
) -> str:
    """Get jsonified basin/subbasin/interbasin geometry including area as a property.

    Parameters
    ----------
    region : dict
        dictionary containing region definition
    data_libs : (list of) str, Path, optional
        One or more paths to data catalog configuration files or names of predefined
        data catalogs. By default the data catalog is initiated without data entries.
        See :py:func:`~hydromt.data_adapter.DataCatalog.from_yml`
        for accepted yaml format.
    hydrography_fn : str
        Name of data source for hydrography data.
    basin_index_fn : str
        Name of data source with basin (bounding box) geometries associated with
            the 'basins' layer of `hydrography_fn`. Only required if the `region` is
            based on a (sub)(inter)basins without a 'bounds' argument.


    Returns
    -------
    geom: str
        Geojson of geodataframe
    """
    if data_libs:
        data_catalog = DataCatalog(data_libs, logger=logger)
    else:
        data_catalog = DataCatalog(logger=logger)
    kind, region = workflows.parse_region(region, logger=logger)
    # NOTE: kind=outlet is deprecated!
    if kind in ["basin", "subbasin", "interbasin", "outlet"]:
        # retrieve global hydrography data (lazy!)
        ds_org = data_catalog.get_rasterdataset(hydrography_fn)
        if "bounds" not in region:
            region.update(basin_index=data_catalog.get_source(basin_index_fn))
        # get basin geometry
        geom, xy = workflows.get_basin_geometry(
            ds=ds_org,
            kind=kind,
            logger=logger,
            **region,
        )

        geom_bbox = geom.geometry.total_bounds
        projected_crs = utm_crs(geom_bbox)
        geom_projected = geom.to_crs(crs=projected_crs)
        geom["area"] = geom_projected["geometry"].area

        return geom.to_json()
    else:
        raise ValueError(
            "Only basin, subbasin, and interbasin are accepted region definitions"
        )
