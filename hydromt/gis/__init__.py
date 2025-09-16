"""Accessors to extend the functionality of xarray structures."""

# required for accessor style documentation
from xarray import DataArray, Dataset  # noqa: F401

from hydromt.gis import flw
from hydromt.gis.gis_utils import parse_crs, utm_crs
from hydromt.gis.raster import RasterDataArray, RasterDataset
from hydromt.gis.raster_utils import (
    full,
    full_from_transform,
    full_like,
    merge,
    spread2d,
)
from hydromt.gis.vector import GeoDataArray, GeoDataset
from hydromt.gis.vector_utils import nearest_merge

__all__ = [
    "RasterDataArray",
    "RasterDataset",
    "GeoDataArray",
    "GeoDataset",
    "full",
    "full_from_transform",
    "full_like",
    "merge",
    "nearest_merge",
    "spread2d",
    "utm_crs",
    "parse_crs",
    "flw",
]
