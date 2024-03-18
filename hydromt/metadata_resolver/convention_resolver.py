"""MetaDataResolver using HydroMT naming conventions."""

from itertools import product
from logging import Logger, getLogger
from string import Formatter
from typing import Any, List, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from hydromt._typing import Bbox, NoDataStrategy, Predicate, TimeRange

from .metadata_resolver import MetaDataResolver

logger: Logger = getLogger(__name__)


class ConventionResolver(MetaDataResolver):
    """MetaDataResolver using HydroMT naming conventions."""

    _uri_placeholders = frozenset({"year", "month", "variable", "zoom_level"})

    def _expand_uri_placeholders(
        self,
        uri: str,
        time_tuple: Optional[Tuple[str, str]] = None,
        variables: Optional[Set[str]] = None,
    ) -> Tuple[str, List[str]]:
        """Expand known placeholders in the URI."""
        keys: list[str] = []

        if "{" in uri:
            uri_expanded = ""
            for literal_text, key, fmt, _ in Formatter().parse(uri):
                uri_expanded += literal_text
                if key is None:
                    continue
                key_str = "{" + f"{key}:{fmt}" + "}" if fmt else "{" + key + "}"
                # remove unused fields
                if key in ["year", "month"] and time_tuple is None:
                    uri_expanded += "*"
                elif key == "variable" and variables is None:
                    uri_expanded += "*"
                # escape unknown fields
                elif key is not None and key not in self._uri_placeholders:
                    uri_expanded = uri_expanded + "{" + key_str + "}"
                else:
                    uri_expanded = uri_expanded + key_str
                    keys.append(key)
            uri = uri_expanded

        return (uri, keys)

    def _get_dates(
        self,
        keys: List[str],
        timerange: TimeRange,
    ) -> pd.PeriodIndex:
        dt: pd.Timedelta = pd.to_timedelta(self.unit_add.get("time", 0), unit="s")
        t_range: pd.DatetimeIndex = pd.to_datetime(list(timerange)) - dt
        freq: str = "m" if "month" in keys else "a"
        dates: pd.PeriodIndex = pd.period_range(*t_range, freq=freq)
        return dates

    def _get_variables(self, variables: List[str]) -> List[str]:
        variables: list[str] = np.atleast_1d(variables).tolist()
        inverse_rename_mapping: dict[str, str] = {v: k for k, v in self.rename.items()}
        vrs: dict[str] = [inverse_rename_mapping.get(var, var) for var in variables]
        return vrs

    def resolve(
        self,
        uri: str,
        *,
        timerange: Optional[TimeRange] = None,
        bbox: Optional[Bbox] = None,
        # TODO: align? -> from RasterDatasetAdapter
        mask: Optional[gpd.GeoDataFrame] = None,
        buffer: float = 0.0,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        zoom_level: int = 0,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
        **kwargs,
    ) -> list[str]:
        """Resolve the placeholders in the URI."""
        uri_expanded, keys = self._expand_uri_placeholders(uri, timerange, variables)
        if timerange:
            dates = self._get_dates(keys, timerange)
        else:
            dates = pd.PeriodIndex(["2023-01-01"], freq="d")
        if variables:
            variables = self._get_variables(variables)
        else:
            variables = [""]
        fmts: list[dict[str, Any]] = list(
            map(
                lambda t: {
                    "year": t[0].year,
                    "month": t[0].month,
                    "variable": t[1],
                    "zoom_level": zoom_level,
                },
                product(dates, variables),
            )
        )
        return [uri_expanded.format(**fmt) for fmt in fmts]
