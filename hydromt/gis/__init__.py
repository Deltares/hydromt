"""Accessors to extend the functionality of xarray structures."""

# required for accessor style documentation
from xarray import DataArray, Dataset  # noqa: F401

from .raster import (
    RasterDataArray,
    RasterDataset,
    full,
    full_from_transform,
    full_like,
)
from .vector import GeoDataArray, GeoDataset

__all__ = [
    "RasterDataArray",
    "RasterDataset",
    "GeoDataArray",
    "GeoDataset",
    "full",
    "full_from_transform",
    "full_like",
]
