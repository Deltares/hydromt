"""DataSource class for the RasterDataSet type."""
from logging import Logger
from typing import Any, Dict, List, Optional

import xarray as xr
from pydantic import ValidationInfo, field_validator, model_validator

from hydromt._typing import Bbox, Geom, TimeRange
from hydromt.data_sources.data_source import DataSource
from hydromt.drivers.rasterdataset_driver import RasterDataSetDriver
from hydromt.drivers.zarr_driver import ZarrDriver

_KNOWN_DRIVERS: Dict[str, RasterDataSetDriver] = {"zarr": ZarrDriver}


def driver_from_str(driver_str: str, **kwargs) -> RasterDataSetDriver:
    """Construct RasterDataSetDriver."""
    if driver_str not in _KNOWN_DRIVERS.keys():
        raise ValueError(
            f"driver {driver_str} not in known RasterDataSetDrivers: {_KNOWN_DRIVERS.keys()}"
        )

    return _KNOWN_DRIVERS[driver_str](**kwargs)


class RasterDataSource(DataSource):
    """DataSource class for the RasterDataSet type."""

    data_type = "RasterDataSet"

    @model_validator(mode="before")
    @classmethod
    def _validate_data_type(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("data_type", "") != "RasterDataSet":
                raise ValueError("'data_type' must be 'RasterDataSet'.")
        return data

    driver: RasterDataSetDriver

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

    zoom_levels: Optional[Dict[int, float]] = None

    def read_data(
        self,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0,
        predicate: str = "intersects",
        timerange: Optional[TimeRange] = None,
        zoom_level: int = 0,
        logger: Optional[Logger] = None,
    ) -> List[xr.Dataset]:
        """
        Read data from this source.

        Data is returned piecewise as a generator, so that users of this class can
        filter the data before taking it all into memory.

        Args:
        """
        uris: List[str] = self.metadata_resolver.resolve(
            self,
            bbox=bbox,
            mask=mask,
            buffer=buffer,
            predicate=predicate,
            timerange=timerange,
            zoom_level=zoom_level,
            logger=logger,
            **self.resolver_kwargs,
        )
        # TODO: think about our raster processing: this (and previous) behavior keeps
        # all in memory. This should be a Generator. To be discussed.
        datasets: List[xr.Dataset] = []
        for uri in uris:
            datasets.append(
                self.driver.read(
                    uri=uri,
                    bbox=bbox,
                    mask=mask,
                    buffer=buffer,
                    crs=self.crs,
                    predicate=predicate,
                    timerange=timerange,
                    zoom_level=zoom_level,
                    **self.driver_kwargs,
                )
            )
        return datasets
