"""Generic DataSource for GeoDataFrames."""

from logging import Logger
from typing import Any, ClassVar, Dict, List, Literal, Optional

import geopandas as gpd
from pydantic import ValidationInfo, field_validator

from hydromt._typing import Bbox, Geom, NoDataStrategy
from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver
from hydromt.drivers.pyogrio_driver import PyogrioDriver

from .data_source import DataSource

# placeholder for proper plugin behaviour later on.
_KNOWN_DRIVERS: Dict[str, GeoDataFrameDriver] = {"pyogrio": PyogrioDriver}


def driver_from_str(driver_str: str, **driver_kwargs) -> GeoDataFrameDriver:
    """Construct GeoDataFrame driver."""
    if driver_str not in _KNOWN_DRIVERS.keys():
        raise ValueError(
            f"driver {driver_str} not in known GeoDataFrameDrivers: {_KNOWN_DRIVERS.keys()}"
        )

    return _KNOWN_DRIVERS[driver_str](**driver_kwargs)


class GeoDataSource(DataSource):
    """
    DataSource for GeoDataFrames.

    Reads and validates DataCatalog entries.
    """

    data_type: ClassVar[Literal["GeoDataFrame"]] = "GeoDataFrame"
    driver: GeoDataFrameDriver

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
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0.0,
        variables: Optional[List[str]] = None,
        predicate: str = "intersects",
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Optional[Logger] = None,
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
            **self.driver_kwargs,
        )
        if len(uris) > 1:
            raise ValueError("GeoDataFrames cannot have more than 1 source URI.")
        return self.driver.read(
            uris, bbox, mask, buffer, self.crs, predicate, logger=logger
        )
