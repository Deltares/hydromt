"""Generic DataSource for GeoDataFrames."""

from logging import Logger

import geopandas as gpd
from pydantic import field_validator

from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver
from hydromt.drivers.pyogrio_driver import PyogrioDriver
from hydromt.metadata_resolvers.resolver import RESOLVERS

# from hydromt.nodata import NoDataStrategy
from .data_source import DataSource

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

    @field_validator("driver")
    @classmethod
    def _check_geodataframe_drivers(cls, driver_str: str) -> str:
        assert driver_str in _KNOWN_DRIVERS
        return driver_str

    def read_data(
        self,
        bbox: list[float] | None = None,
        mask: gpd.GeoDataFrame | None = None,
        buffer: float = 0.0,
        variables: list[str] | None = None,
        predicate: str = "intersects",
        logger: Logger | None = None,
    ) -> gpd.GeoDataFrame:
        """Use initialize driver to read data."""
        # uris: list[str] = PlaceHolderURI(self.uri, variables=variables).expand(self)
        uris: list[str] = RESOLVERS[self.metadata_resolver].resolve(
            self,
            bbox=bbox,
            geom=mask,
            buffer=buffer,
            predicate=predicate,
            variables=variables,
        )
        if len(uris) > 1:
            raise ValueError("GeoDataFrames cannot have more than 1 source URI.")
        uri = uris[0]
        gdf_driver: GeoDataFrameDriver = driver_from_str(
            self.driver, **self.driver_kwargs
        )
        return gdf_driver.read(
            uri, bbox, mask, buffer, self.crs, predicate, logger=logger
        )
