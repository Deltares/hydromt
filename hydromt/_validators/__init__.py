"""Pydantic models for validation of various hydromt internal components."""

from enum import StrEnum

from hydromt._validators.data_catalog_v0x import (
    DataCatalogV0Item,
    DataCatalogV0ItemMetadata,
    DataCatalogV0MetaData,
    DataCatalogV0Validator,
)
from hydromt._validators.data_catalog_v1x import (
    DataCatalogV1Item,
    DataCatalogV1ItemMetadata,
    DataCatalogV1MetaData,
    DataCatalogV1Validator,
)
from hydromt._validators.model_config import HydromtModelSetup, HydromtModelStep
from hydromt._validators.region import (
    BoundingBoxRegion,
    PathRegion,
    Region,
    validate_region,
)

__all__ = [
    "DataCatalogV1Item",
    "DataCatalogV1ItemMetadata",
    "DataCatalogV1MetaData",
    "DataCatalogV1Validator",
    "DataCatalogV0Item",
    "DataCatalogV0ItemMetadata",
    "DataCatalogV0MetaData",
    "DataCatalogV0Validator",
    "BoundingBoxRegion",
    "PathRegion",
    "Region",
    "validate_region",
    "HydromtModelStep",
    "HydromtModelSetup",
    "Format",
]


class Format(StrEnum):
    v0 = "v0"
    v1 = "v1"
