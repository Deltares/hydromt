"""Generic DataSource for GeoDataFrames."""

from logging import Logger
from typing import Any, List, Optional

import geopandas as gpd
from pydantic import ValidationInfo, field_validator, model_validator

from hydromt._typing import NoDataStrategy
from hydromt._typing.type_def import Predicate
from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver
from hydromt.drivers.pyogrio_driver import PyogrioDriver

from .data_source import DataSource

# placeholder for proper plugin behaviour later on.
_KNOWN_DRIVERS: dict[str, GeoDataFrameDriver] = {"pyogrio": PyogrioDriver}


def driver_from_str(driver_str: str, **driver_kwargs) -> GeoDataFrameDriver:
    """Construct GeoDataFrame driver."""
    if driver_str not in _KNOWN_DRIVERS.keys():
        raise ValueError(
            f"driver {driver_str} not in known GeoDataFrameDrivers: {_KNOWN_DRIVERS.keys()}"
        )

    return _KNOWN_DRIVERS[driver_str](**driver_kwargs)


class GeoDataFrameDataSource(DataSource):
    """
    DataSource for GeoDataFrames.

    Reads and validates DataCatalog entries.
    """

    data_type = "GeoDataFrame"
    driver: GeoDataFrameDriver

    @model_validator(mode="before")
    @classmethod
    def _validate_data_type(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("data_type", "") != "GeoDataFrame":
                raise ValueError("'data_type' must be 'GeoDataFrame'.")
        return data

    @field_validator("driver", mode="before")
    @classmethod
    def _check_geodataframe_drivers(cls, v: Any, info: ValidationInfo) -> str:
        if isinstance(v, str):
            if v not in _KNOWN_DRIVERS:
                raise ValueError(f"unknown driver '{v}'")
            return driver_from_str(v, **info.data.get("driver_kwargs"))
        elif hasattr(v, "read"):  # driver duck-typing
            return v
        else:
            raise ValueError(f"unknown driver type: {str(v)}")

    def read_data(
        self,
        region: Optional[gpd.GeoDataFrame] = None,
        variables: Optional[List[str]] = None,
        predicate: Predicate = "intersects",
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Optional[Logger] = None,
    ) -> gpd.GeoDataFrame:
        """Use initialize driver to read data."""
        uris: list[str] = self.metadata_resolver.resolve(
            self,
            region=region,
            predicate=predicate,
            variables=variables,
            handle_nodata=handle_nodata,
        )
        if len(uris) > 1:
            raise ValueError("GeoDataFrames cannot have more than 1 source URI.")
        uri = uris[0]
        return self.driver.read(uri, region=region, predicate=predicate, logger=logger)
