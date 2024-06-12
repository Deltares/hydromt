"""MetaDataResolver using HydroMT naming conventions."""

from functools import reduce
from itertools import chain, product
from logging import Logger, getLogger
from re import compile as compile_regex
from re import error as regex_error
from string import Formatter
from typing import Any, Dict, Iterable, List, Optional, Pattern, Set, Tuple

import pandas as pd
from fsspec import AbstractFileSystem
from fsspec.core import split_protocol

from hydromt._typing import (
    Geom,
    NoDataStrategy,
    TimeRange,
    ZoomLevel,
    exec_nodata_strat,
)
from hydromt._utils.unused_kwargs import _warn_on_unused_kwargs

from .metadata_resolver import MetaDataResolver

logger: Logger = getLogger(__name__)


class ConventionResolver(MetaDataResolver):
    """MetaDataResolver using HydroMT naming conventions."""

    _uri_placeholders = frozenset({"year", "month", "variable", "name"})
    name = "convention"

    def _expand_uri_placeholders(
        self,
        uri: str,
        time_tuple: Optional[Tuple[str, str]] = None,
        variables: Optional[List[str]] = None,
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
        """Obtain the dates the user is searching for."""
        t_range: pd.DatetimeIndex = pd.to_datetime(list(time_range))
        freq: str = "M" if "month" in keys else "Y"
        dates: pd.PeriodIndex = pd.period_range(*t_range, freq=freq)
        return dates

    def _resolve_wildcards(
        self, uris: Iterable[str], fs: AbstractFileSystem
    ) -> Set[str]:
        """Expand on the wildcards in the uris based on the filesystem."""

        def split_and_glob(uri: str) -> Tuple[Optional[str], List[str]]:
            protocol, _ = split_protocol(uri)
            return (protocol, fs.glob(uri))

        def maybe_unstrip_protocol(
            pair: Tuple[Optional[str], Iterable[str]],
        ) -> Iterable[str]:
            if pair[0] is not None:
                return map(
                    lambda uri: fs.unstrip_protocol(uri)
                    if not uri.startswith(pair[0])
                    else uri,
                    pair[1],
                )
            else:
                return pair[1]

        return set(
            chain.from_iterable(  # flatten result
                map(
                    lambda uri_pair: maybe_unstrip_protocol(
                        uri_pair
                    ),  # keep protocol in uri if present
                    reduce(
                        lambda uri_res, uri: uri_res + [split_and_glob(uri)], uris, []
                    ),
                )
            )
        )

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
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Resolve the placeholders in the URI using naming conventions.

        Parameters
        ----------
        uri : str
            Unique Resource Identifier
        fs : AbstractFileSystem
            fsspec filesystem used to resolve wildcards in the uri
        time_range : Optional[TimeRange], optional
            left-inclusive start end time of the data, by default None
        mask : Optional[Geom], optional
            A geometry defining the area of interest, by default None
        zoom_level : Optional[ZoomLevel], optional
            zoom_level of the dataset, by default None
        variables : Optional[List[str]], optional
            Names of variables to return, or all if None, by default None
        handle_nodata : NoDataStrategy, optional
            how to react when no data is found, by default NoDataStrategy.RAISE
        logger : Logger, optional
            logger to use, by default logger
        options : Optional[Dict[str, Any]], optional
            extra options for this resolver, by default None

        Returns
        -------
        List[str]
            a list of expanded uris

        Raises
        ------
        NoDataException
            when no data is found and `handle_nodata` is `NoDataStrategy.RAISE`
        """
        _warn_on_unused_kwargs(
            self.__class__.__name__, {"mask": mask, "zoom_level": zoom_level}, logger
        )

        uri_expanded, keys, _ = self._expand_uri_placeholders(
            uri, time_range, variables
        )
        if time_range:
            dates = self._get_dates(keys, time_range)
        else:
            dates = pd.PeriodIndex(["1970-01-01"], freq="d")  # fill any valid value
        if not variables:
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
            exec_nodata_strat(
                f"resolver '{self.name}' found no files.",
                strategy=handle_nodata,
                logger=logger,
            )
            return []  # if ignore

        return uris
