"""MetaDataResolver using HydroMT naming conventions."""
from itertools import product
from string import Formatter
from typing import Any

import numpy as np
import pandas as pd

from .metadata_resolver import MetaDataResolver


class ConventionResolver(MetaDataResolver):
    """MetaDataResolver using HydroMT naming conventions."""

    _uri_placeholders = {"year", "month", "variable"}

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
        self, keys: list[str], time_tuple: tuple[str, str]
    ) -> pd.PeriodIndex:
        dt: pd.Timedelta = pd.to_timedelta(
            self.source.unit_add.get("time", 0), unit="s"
        )
        t_range: pd.DatetimeIndex = pd.to_datetime(list(time_tuple)) - dt
        freq: str = "m" if "month" in keys else "a"
        dates: pd.PeriodIndex = pd.period_range(*t_range, freq=freq)
        return dates

    def _get_variables(self, variables: list[str]) -> list[str]:
        variables: list[str] = np.atleast_1d(variables).tolist()
        mv_inv: dict[str, str] = {v: k for k, v in self.source.rename.items()}
        vrs: dict[str] = [mv_inv.get(var, var) for var in variables]
        return vrs

    def resolve_uri(self, uri: str, **kwargs) -> list[str]:
        """Resolve the placeholders in the URI."""
        time_tuple: tuple[str, str] | None = kwargs.get("time_tuple")
        variables: set[str, str] | None = kwargs.get("variables")
        uri_expanded, keys = self._expand_uri_placeholders(uri, time_tuple, variables)
        dates = self._get_dates(keys, time_tuple)
        vrs = self._get_variables(variables)
        fmts: list[dict[str, Any]] = list(
            map(
                lambda d, var: {"year": d.year, "month": d.month, "variable": var},
                product(dates, vrs),
            )
        )
        return [uri_expanded.format(fmt) for fmt in fmts]
