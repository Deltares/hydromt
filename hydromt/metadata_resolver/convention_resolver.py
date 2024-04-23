"""MetaDataResolver using HydroMT naming conventions."""

from functools import reduce
from itertools import product
from logging import Logger, getLogger
from re import compile as compile_regex
from re import error as regex_error
from string import Formatter
from typing import Any, Dict, Iterable, List, Optional, Pattern, Set, Tuple

import numpy as np
import pandas as pd
from fsspec import AbstractFileSystem

from hydromt._typing import Geom, NoDataStrategy, TimeRange, ZoomLevel
from hydromt.utils.unused_kwargs import unused_kwargs_method, warn_on_unused_kwargs

from .metadata_resolver import MetaDataResolver

logger: Logger = getLogger(__name__)


class ConventionResolver(MetaDataResolver):
    """MetaDataResolver using HydroMT naming conventions."""

    _uri_placeholders = frozenset({"year", "month", "variable", "name"})

    def _expand_uri_placeholders(
        self,
        uri: str,
        time_tuple: Optional[Tuple[str, str]] = None,
        variables: Optional[Set[str]] = None,
    ) -> Tuple[str, List[str], Pattern[str]]:
        """Expand known placeholders in the URI."""
        keys: list[str] = []
        pattern: str = ""

        if "{" in uri:
            uri_expanded = ""
            for literal_text, key, fmt, _ in Formatter().parse(uri):
                uri_expanded += literal_text
                pattern += literal_text
                if key is None:
                    continue
                pattern += "(.*)"
                key_str = "{" + f"{key}:{fmt}" + "}" if fmt else "{" + key + "}"
                # remove unused fields
                if key in ["year", "month"] and time_tuple is None:
                    uri_expanded += "*"
                elif key == "variable" and variables is None:
                    uri_expanded += "*"
                elif key == "name":
                    uri_expanded += "*"
                # escape unknown fields
                elif key is not None and key not in self._uri_placeholders:
                    uri_expanded = uri_expanded + "{" + key_str + "}"
                else:
                    uri_expanded = uri_expanded + key_str
                    keys.append(key)
            uri = uri_expanded

        # darn windows paths creating invalid escape sequences grrrrr
        try:
            regex = compile_regex(pattern)
        except regex_error:
            # try it as raw path if regular string fails
            regex = compile_regex(pattern.encode("unicode_escape").decode())

        return (uri, keys, regex)

    def _get_dates(
        self,
        keys: List[str],
        time_range: TimeRange,
    ) -> pd.PeriodIndex:
        dt: pd.Timedelta = pd.to_timedelta(self.unit_add.get("time", 0), unit="s")
        t_range: pd.DatetimeIndex = pd.to_datetime(list(time_range)) - dt
        freq: str = "M" if "month" in keys else "a"
        dates: pd.PeriodIndex = pd.period_range(*t_range, freq=freq)
        return dates

    def _get_variables(self, variables: List[str]) -> List[str]:
        variables: list[str] = np.atleast_1d(variables).tolist()
        inverse_rename_mapping: dict[str, str] = {v: k for k, v in self.rename.items()}
        vrs: dict[str] = [inverse_rename_mapping.get(var, var) for var in variables]
        return vrs

    def _resolve_wildcards(
        self, uris: Iterable[str], fs: AbstractFileSystem
    ) -> Set[str]:
        return set(reduce(lambda uri_res, uri: uri_res + fs.glob(uri), uris, []))

    @unused_kwargs_method(unused_kwargs=["mask", "zoom_level"], logger=logger)
    def resolve(
        self,
        uri: str,
        fs: AbstractFileSystem,
        *,
        time_range: Optional[TimeRange] = None,
        mask: Optional[Geom] = None,
        zoom_level: Optional[ZoomLevel] = None,
        variables: Optional[List[str]] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Logger = logger,
    ) -> List[str]:
        """Resolve the placeholders in the URI."""
        warn_on_unused_kwargs(
            self, dict(zip(("mask", "zoom_level"), (mask, zoom_level))), logger
        )

        uri_expanded, keys, _ = self._expand_uri_placeholders(
            uri, time_range, variables
        )
        if time_range:
            dates = self._get_dates(keys, time_range)
        else:
            dates = pd.PeriodIndex(["2023-01-01"], freq="d")  # fill any valid value
        if variables:
            variables = self._get_variables(variables)
        else:
            variables = [""]  # fill any valid value
        fmts: Iterable[Dict[str, Any]] = map(
            lambda t: {
                "year": t[0].year,
                "month": t[0].month,
                "variable": t[1],
            },
            product(dates, variables),
        )
        uris: List[str] = list(
            self._resolve_wildcards(
                map(lambda fmt: uri_expanded.format(**fmt), fmts), fs
            )
        )
        if not uris:
            raise FileNotFoundError(f"No files found for: {uri_expanded}")
        return uris
