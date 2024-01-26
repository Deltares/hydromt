"""Generic DataSource for GeoDataFrames."""

from logging import Logger
from typing import Any

import geopandas as gpd
from pydantic import ValidationInfo, field_validator

from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver
from hydromt.drivers.pyogrio_driver import PyogrioDriver
from hydromt.nodata import NoDataStrategy

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

    driver: GeoDataFrameDriver

    @field_validator("driver", mode="before")
    @classmethod
    def _check_geodataframe_drivers(cls, v: Any, info: ValidationInfo) -> str:
        if isinstance(v, str):
            if v not in _KNOWN_DRIVERS:
                raise ValueError(f"unknown driver '{v}'")
            return driver_from_str(v, **info.data.get("driver_kwargs"))
        elif isinstance(v, GeoDataFrameDriver):
            return v
        else:
            raise ValueError(f"unknown driver type: {str(v)}")

    def read_data(
        self,
        bbox: list[float] | None = None,
        mask: gpd.GeoDataFrame | None = None,
        buffer: float = 0.0,
        variables: list[str] | None = None,
        predicate: str = "intersects",
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger | None = None,
    ) -> gpd.GeoDataFrame:
        """Use initialize driver to read data."""
        uris: list[str] = self.metadata_resolver.resolve(
            self,
            bbox=bbox,
            geom=mask,
            buffer=buffer,
            predicate=predicate,
            variables=variables,
            handle_nodata=handle_nodata,
        )
        if len(uris) > 1:
            raise ValueError("GeoDataFrames cannot have more than 1 source URI.")
        uri = uris[0]
        return self.driver.read(
            uri, bbox, mask, buffer, self.crs, predicate, logger=logger
        )
