"""MetaDataResolver using HydroMT naming conventions."""
from itertools import product
from string import Formatter
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from hydromt.data_sources.data_source import DataSource
from hydromt.nodata import NoDataStrategy
from hydromt.typing import Bbox, TimeRange

from .metadata_resolver import MetaDataResolver


class ConventionResolver(MetaDataResolver):
    """MetaDataResolver using HydroMT naming conventions."""

    _uri_placeholders = {"year", "month", "variable", "zoom_level"}

    def _expand_uri_placeholders(
        self, uri: str, time_tuple: tuple[str, str] | None, variables: set[str] | None
    ) -> tuple[str, list[str]]:
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

        return (uri_expanded, keys)

    def _get_dates(
        self,
        keys: list[str],
        timerange: TimeRange,
    ) -> pd.PeriodIndex:
        dt: pd.Timedelta = pd.to_timedelta(
            self.source.unit_add.get("time", 0), unit="s"
        )
        t_range: pd.DatetimeIndex = pd.to_datetime(list(timerange)) - dt
        freq: str = "m" if "month" in keys else "a"
        dates: pd.PeriodIndex = pd.period_range(*t_range, freq=freq)
        return dates

    def _get_variables(self, variables: list[str]) -> list[str]:
        variables: list[str] = np.atleast_1d(variables).tolist()
        mv_inv: dict[str, str] = {v: k for k, v in self.source.rename.items()}
        vrs: dict[str] = [mv_inv.get(var, var) for var in variables]
        return vrs

    def resolve_uri(
        self,
        uri: str,
        source: DataSource,
        *,
        timerange: TimeRange | None = None,
        bbox: Bbox | None = None,
        # TODO: align? -> from RasterDataSetAdapter
        geom: gpd.GeoDataFrame | None = None,
        buffer: float = 0.0,
        predicate: str = "intersects",
        variables: list[str] | None = None,
        zoom_level: int = 0,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
    ) -> list[str]:
        """Resolve the placeholders in the URI."""
        uri_expanded, keys = self._expand_uri_placeholders(uri, timerange, variables)
        if timerange:
            dates = self._get_dates(keys, timerange)
        else:
            dates = pd.PeriodIndex(data=np.zeros(1))
        if variables:
            variables = self._get_variables(variables)
        vrs = self._get_variables(variables)
        fmts: list[dict[str, Any]] = list(
            map(
                lambda d, var: {
                    "year": d.year,
                    "month": d.month,
                    "variable": var,
                    "zoom_level": zoom_level,
                },
                product(dates, vrs),
            )
        )
        return [uri_expanded.format(fmt) for fmt in fmts]
