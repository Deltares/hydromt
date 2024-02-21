"""A class to handle all functionalities to do with model regions."""

from logging import Logger, getLogger
from os.path import exists
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import xarray as xr
from geopandas import GeoDataFrame, GeoSeries
from numpy._typing import NDArray
from pyproj import CRS

from hydromt._typing.model_mode import ModelMode
from hydromt._typing.type_def import BasinIdType, StrPath
from hydromt.models._region.specifyers import RegionSpecifyer

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
        if exists(path) and not mode.is_writing_mode():
            raise ValueError(
                "not in write mode, therefore cannot write region to file."
            )

        if self._data is None:
            self.construct()
            self._data = cast(GeoDataFrame, self._data)

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
        """Parse the user provided dictionary to one pydantic can parse. Note that the kind MUST be the first key."""
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
        elif kind == "geom":
            if isinstance(value, (GeoDataFrame, GeoSeries)):
                flat_region_dict = {"kind": "geom_data", "geom": value}
            elif isinstance(value, Path):
                flat_region_dict = {"kind": "geom_file", "path": Path(value)}
            elif isinstance(value, str):
                flat_region_dict = {"kind": "geom_catalog", "source": value}
        elif kind == "grid":
            if isinstance(value, Path):
                flat_region_dict = {
                    "kind": "grid_path",
                    "path": Path(value),
                }
            if isinstance(value, str):
                flat_region_dict = {
                    "kind": "grid_catalog",
                    "source": value,
                }
            elif isinstance(value, (xr.Dataset, xr.DataArray)):
                flat_region_dict = {"kind": "grid_data", "data": value}

        elif kind == "mesh":
            flat_region_dict = {"kind": "mesh", "path": Path(value)}
        elif kind == "model":
            flat_region_dict = {"kind": "model", "path": Path(value)}
        elif kind == "basin":
            if isinstance(value, BasinIdType):
                flat_region_dict = {"kind": "basin_id", "id": value}
            elif isinstance(value, list):
                if isinstance(value[0], BasinIdType):
                    flat_region_dict = {
                        "kind": "basin_ids",
                        "ids": value,
                    }
                elif isinstance(value[0], float) and len(value) == 2:
                    flat_region_dict = {
                        "kind": "basin_xy",
                        "x": value[0],
                        "y": value[1],
                    }
                elif isinstance(value[0], float) and len(value) == 4:
                    flat_region_dict = {
                        "kind": "basin_bbox",
                        "xmin": value[0],
                        "ymin": value[1],
                        "xmax": value[2],
                        "ymax": value[3],
                        **region,  # add variables and outlets if they exist
                    }
                elif isinstance(value[0], list) and len(value) == 2:
                    flat_region_dict = {
                        "kind": "basin_xys",
                        "xs": value[0],
                        "ys": value[1],
                    }
                else:
                    raise RuntimeError(
                        f"unreachable basin spec while parsing: {region}"
                    )

        elif kind == "subbasin":
            if isinstance(value, Path):
                flat_region_dict = {
                    "kind": "subbasin_geom",
                    "path": value,
                    **region,  # add variables and outlets if they exist
                }
            elif isinstance(value[0], float) and len(value) == 2:
                flat_region_dict = {
                    "kind": "subbasin_xy",
                    "x": value[0],
                    "y": value[1],
                }
            elif isinstance(value[0], float) and len(value) == 4:
                flat_region_dict = {
                    "kind": "subbasin_bbox",
                    "xmin": value[0],
                    "ymin": value[1],
                    "xmax": value[2],
                    "ymax": value[3],
                    **region,  # add variables and outlets if they exist
                }
            elif isinstance(value[0], list) and len(value) == 2:
                flat_region_dict = {
                    "kind": "subbasin_xys",
                    "xs": value[0],
                    "ys": value[1],
                    **region,
                }
            else:
                raise ValueError(
                    f"could not understand subbasin specification: {kind:value, **region}"
                )
        elif kind == "interbasin":
            if isinstance(value, Path):
                flat_region_dict = {
                    "kind": "interbasin_geom",
                    "path": value,
                    **region,  # add variables and outlets if they exist
                }
            elif isinstance(value, list):
                if "xy" in region:
                    flat_region_dict = {
                        "kind": "interbasin_bbox_xy",
                        "xmin": value[0],
                        "ymin": value[1],
                        "xmax": value[2],
                        "ymax": value[3],
                        **region,  # add variables and outlets if they exist
                    }
                else:
                    flat_region_dict = {
                        "kind": "interbasin_bbox_var",
                        "xmin": value[0],
                        "ymin": value[1],
                        "xmax": value[2],
                        "ymax": value[3],
                        **region,  # add variables and outlets if they exist
                    }
            elif isinstance(value, GeoDataFrame):
                flat_region_dict = {
                    "kind": "interbasin_geom",
                    "data": value,
                    **region,  # add variables and outlets if they exist
                }
        else:
            raise ValueError(f"Unknown region kind: {kind}")

        return RegionSpecifyer(spec=flat_region_dict)  # type: ignore
