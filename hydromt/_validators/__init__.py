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
    V0 = (0,)
    V1 = (1,)

    # just a convenience function
    # with some nicer error messages
    @staticmethod
    def from_str(s: str) -> "Format":
        try:
            return Format[s.upper().strip()]
        except KeyError as e:
            raise TypeError(
                f"{e} is not a known valid Format, options are {list(Format.__members__.keys())}"
            )
