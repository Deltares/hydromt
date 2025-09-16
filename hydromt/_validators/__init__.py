"""Pydantic models for validation of various hydromt internal components."""

from enum import Enum

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


class Format(Enum):
    v0 = 0
    v1 = 1

    # just a convenience function
    # with some nicer error messages
    @classmethod
    def from_str(cls, s: str) -> "Format":
        try:
            return cls[s.strip().lower()]
        except KeyError as e:
            raise ValueError(
                f"{e} is not a known valid Format, options are {list(cls.__members__.keys())}"
            )
