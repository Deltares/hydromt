"""Accessors to extend the functionality of xarray structures."""

# required for accessor style documentation
from xarray import DataArray, Dataset  # noqa: F401

from hydromt.gis import flw
from hydromt.gis._gis_utils import utm_crs
from hydromt.gis.raster import (
    RasterDataArray,
    RasterDataset,
    full,
    full_from_transform,
    full_like,
)
from hydromt.gis.vector import GeoDataArray, GeoDataset

__all__ = [
    "RasterDataArray",
    "RasterDataset",
    "GeoDataArray",
    "GeoDataset",
    "full",
    "full_from_transform",
    "full_like",
    "utm_crs",
    "flw",
]
