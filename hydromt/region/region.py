"""A class to handle all functionalities to do with model regions."""

from logging import Logger, getLogger
from os.path import exists
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from numpy._typing import NDArray
from pyproj import CRS

from hydromt._typing.model_mode import ModelMode
from hydromt._typing.type_def import StrPath
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

    @property
    def data(self) -> GeoDataFrame:
        """Initialise the data of the region if it isn't already and then return it."""
        if self._data is None:
            self.construct()

        return cast(GeoDataFrame, self._data)

    @property
    def bounds(self) -> NDArray[Any]:
        """A shortcut to the bounds of the region."""
        if self._data is None:
            self.construct()

        return cast(GeoDataFrame, self._data).total_bounds

    @property
    def crs(self) -> CRS:
        """A shortcut to the CRS of the region."""
        if self._data is None:
            self.construct()

        return cast(GeoDataFrame, self._data).crs

    def write(
        self,
        path: StrPath,
        mode: ModelMode = ModelMode.WRITE,
        geopandas_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Write the geometry to a file."""
        if not mode.is_writing_mode():
            raise ValueError("Cannot write region when not in writing mode.")

        # intentionally not using .is_override() because it makes no sense to append multiple regions
        # as they might be in different CRSs for example
        if exists(path) and mode != ModelMode.FORCED_WRITE:
            raise ValueError(
                f"Attempted to write geom at {path}, but the file already exist and mode was not in forced override mode"
            )

        if self._data is None:
            raise ValueError(
                "Region is not yet initialised. use the construct() method."
            )
        else:
            if geopandas_kwargs is not None:
                kwargs = geopandas_kwargs
            else:
                kwargs = {}
            self._data.to_file(path, **kwargs)

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
