"""Abstract DataSource class."""
from itertools import product
from string import Formatter
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from hydromt import DataCatalog
from hydromt.metadata_resolvers.resolver import RESOLVERS


class PlaceHolderURI:
    """Data URI with placeholder for resolving naming conventions."""

    _uri_placeholders = {"year", "month", "variable"}

    def __init__(
        self,
        uri: str,
        time_tuple: tuple[str, str] | None = None,
        variables: list[str] | None = None,
    ):
        keys: list[str] = []

        if "{" in self.uri:
            uri = ""
            for literal_text, key, fmt, _ in Formatter().parse(self.uri):
                uri += literal_text
                if key is None:
                    continue
                key_str = "{" + f"{key}:{fmt}" + "}" if fmt else "{" + key + "}"
                # remove unused fields
                if key in ["year", "month"] and time_tuple is None:
                    uri += "*"
                elif key == "variable" and variables is None:
                    uri += "*"
                # escape unknown fields
                elif key is not None and key not in self._uri_placeholders:
                    uri = uri + "{" + key_str + "}"
                else:
                    uri = uri + key_str
                    keys.append(key)

        self._uri = uri
        self._keys = keys
        self._time_tuple = time_tuple
        self._variables = variables
        self._fmt = []

    @property
    def uri(self) -> str:
        """URI with placeholders."""
        return self._uri

    def _get_dates(self, unit_add: dict[str, Any]) -> pd.PeriodIndex:
        dt: pd.Timedelta = pd.to_timedelta(unit_add.get("time", 0), unit="s")
        t_range: pd.DatetimeIndex = pd.to_datetime(list(self._time_tuple)) - dt
        freq: str = "m" if "month" in self._keys else "a"
        dates: pd.PeriodIndex = pd.period_range(*t_range, freq=freq)
        return dates

    def _get_variables(self, rename: dict[str, str]) -> list[str]:
        variables: list[str] = np.atleast_1d(self._variables).tolist()
        mv_inv: dict[str, str] = {v: k for k, v in rename.items()}
        vrs: dict[str] = [mv_inv.get(var, var) for var in variables]
        return vrs

    def get_format_maps(self, source: "DataSource") -> list[dict[str, Any]]:
        """
        Obtain format maps from the data source.

        The result is a list of dictionaries that format the uri and thus solve the naming
        conventions embedded in the uri.

        Parameters
        ----------
        source: DataSource
            the DataSource to expand the URI for.

        Returns
        -------
        list[dict[str, Any]]
            A list of maps that are used to format the URI.
        """
        dates = self._get_dates(source.unit_add)
        vrs = self._get_variables(source.rename)
        return list(
            map(
                lambda d, var: {"year": d.year, "month": d.month, "variable": var},
                product(dates, vrs),
            )
        )

    def expand(self, source: "DataSource") -> list[str]:
        """
        Expand the uri using all known placeholders and the data source.

        Parameters
        ----------
        source: DataSource
            the DataSource to expand the URI for.

        Returns
        -------
        list[str]
            The expanded set of URIs.
        """
        fmts: list[dict[str, Any]] = self.get_format_maps(source)
        return [self.uri.format(fmt) for fmt in fmts]


class DataSource(BaseModel):
    """
    A DataSource is a parsed section of a DataCatalog.

    The DataSource, specific for a data type within HydroMT, is responsible for
    validating the input from the DataCatalog, to
    ensure the workflow fails as early as possible. A DataSource has information on
    the driver that the data should be read with, and is responsible for initializing
    this driver.
    """

    @classmethod
    def from_catalog(
        cls,
        catalog: DataCatalog,
        key: str,
        provider: str | None = None,
        version: str | None = None,
    ) -> "DataSource":
        """Create Data source from DataCatalog."""
        return cls.model_validate(catalog.get_source(key, provider, version))

    def _validate_metadata_resolver(cls, v: Any):
        assert isinstance(v, str), "metadata_resolver should be string."
        assert v in RESOLVERS, f"unknown MetaDataResolver: '{v}'."
        RESOLVERS.get(v)

    version: str | None = Field(default=None)
    provider: str | None = Field(default=None)
    driver: str
    metadata_resolver: str
    driver_kwargs: dict[str, Any] = Field(default_factory=dict)
    unit_add: dict[str, Any] = Field(default_factory=dict)
    unit_mult: dict[str, Any] = Field(default_factory=dict)
    rename: dict[str, str] = Field(default_factory=dict)
    extent: dict[str, Any] = Field(default_factory=dict)  # ?
    meta: dict[str, Any] = Field(default_factory=dict)
    uri: str
    crs: int | None = Field(default=None)
