"""A class to handle all functionalities to do with model regions."""

from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pyproj import CRS
from shapely import box

from hydromt._typing import Bbox, Geom, Predicate
from hydromt.region._specifyers import RegionSpecifyer

logger = getLogger(__name__)


class Region:
    """A class to handle all operations to do with the model region."""

    def __init__(
        self,
        region_dict: Dict[str, Any],
        logger: Logger = logger,
    ):
        self._dict: Dict[str, Any] = region_dict
        self._data: Optional[GeoDataFrame] = None
        self._spec = self._parse_region(region_dict, logger=logger)
        self.logger = logger

    def write(self, path):
        """Write the geometry to a file."""
        if self._data is None:
            raise ValueError(
                "Region is not yet initialised. use the construct() method."
            )
        else:
            self._data.to_file(path)

    def construct(self) -> GeoDataFrame:
        """Calculate the actual geometry based on the specification."""
        self._data = self._spec.construct()
        return self._data

    @staticmethod
    def _parse_region(
        region: Dict[str, Any],
        logger=logger,
    ) -> RegionSpecifyer:
        # popitem returns last inserted, we want first
        kind = next(iter(region.keys()))
        value = region.pop(kind)
        if isinstance(value, np.ndarray):
            value = value.tolist()

        flat_region_dict: Dict[str, Any] = {}
        if kind == "bbox":
            flat_region_dict = {
                "kind": "bbox",
                **dict(zip(["xmin", "ymin", "xmax", "ymax"], value)),
            }
        elif isinstance(value, (GeoDataFrame, GeoSeries)):
            flat_region_dict = {"kind": kind, "geom": value}
        elif isinstance(value, (Path, str)):
            if kind == "geom":
                flat_region_dict = {"kind": "geom_file", "path": Path(value)}
            else:
                flat_region_dict = {"kind": kind, "path": Path(value)}
        else:
            raise ValueError(f"Unknown region kind: {kind}")

        if "buffer" in region:
            flat_region_dict["buffer"] = region["buffer"]

        return RegionSpecifyer(spec=flat_region_dict)  # type: ignore
