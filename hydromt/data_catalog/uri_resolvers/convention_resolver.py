"""URIResolver using HydroMT naming conventions."""

import logging
from functools import reduce
from itertools import chain, product
from typing import Any, Iterable, Optional

import pandas as pd
from fsspec.core import split_protocol

from hydromt._utils.naming_convention import _expand_uri_placeholders
from hydromt.data_catalog.uri_resolvers.uri_resolver import URIResolver
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.gis.gis_utils import zoom_to_overview_level
from hydromt.typing import (
    Geom,
    SourceMetadata,
    TimeRange,
    Zoom,
)

logger = logging.getLogger(__name__)


class ConventionResolver(URIResolver):
    """URIDataResolver using HydroMT naming conventions."""

    _uri_placeholders = frozenset(
        {"year", "month", "variable", "name", "overview_level"}
    )
    name = "convention"

    def _get_dates(
        self,
        keys: list[str],
        time_range: TimeRange,
    ) -> pd.PeriodIndex:
        """Obtain the dates the user is searching for."""
        t_range: pd.DatetimeIndex = pd.to_datetime([time_range.start, time_range.end])
        freq: str = "M" if "month" in keys else "Y"
        dates: pd.PeriodIndex = pd.period_range(*t_range, freq=freq)
        return dates

    def _resolve_wildcards(self, uris: Iterable[str]) -> set[str]:
        """Expand on the wildcards in the uris based on the filesystem."""

        def split_and_glob(uri: str) -> tuple[Optional[str], list[str]]:
            protocol, _ = split_protocol(uri)
            return (protocol, self.filesystem.get_fs().glob(uri))

        def maybe_unstrip_protocol(
            pair: tuple[Optional[str], Iterable[str]],
        ) -> Iterable[str]:
            if pair[0] is not None:
                return map(
                    lambda uri: self.filesystem.get_fs().unstrip_protocol(uri)
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
        *,
        time_range: Optional[TimeRange] = None,
        mask: Optional[Geom] = None,
        zoom: Optional[Zoom] = None,
        variables: Optional[list[str]] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> list[str]:
        """Resolve the placeholders in the URI using naming conventions.

        Parameters
        ----------
        uri : str
            Unique Resource Identifier
        time_range : Optional[TimeRange], optional
            left-inclusive start end time of the data, by default None
        mask : Optional[Geom], optional
            A geometry defining the area of interest, by default None
        zoom: Optional[Zoom], optional
            zoom of the dataset, by default None
        variables : Optional[list[str]], optional
            Names of variables to return, or all if None, by default None
        metadata: Optional[SourceMetadata], optional
            DataSource metadata.
        handle_nodata : NoDataStrategy, optional
            how to react when no data is found, by default NoDataStrategy.RAISE

        Returns
        -------
        list[str]
            a list of expanded uris

        Raises
        ------
        NoDataException
            when no data is found and `handle_nodata` is `NoDataStrategy.RAISE`
        """
        logger.debug(f"attempting to resolve uri: {uri}")

        if metadata is None:
            metadata: SourceMetadata = SourceMetadata()

        uri_expanded, keys, _ = _expand_uri_placeholders(
            uri,
            placeholders=self._uri_placeholders,
            time_range=time_range,
            variables=variables,
        )
        if time_range:
            dates = self._get_dates(keys, time_range)
        else:
            dates = pd.PeriodIndex(["1970-01-01"], freq="d")  # fill any valid value
        if not variables:
            variables = [""]  # fill any valid value
        if zoom:
            zls_dict: Optional[dict[int, float]] = None
            if metadata is not None and hasattr(metadata, "zls_dict"):
                zls_dict = metadata.zls_dict
            crs = None if metadata is None else metadata.crs
            overview_level: int = zoom_to_overview_level(zoom, mask, zls_dict, crs) or 0
        else:
            overview_level = 0  # fill any valid value

        fmts: Iterable[dict[str, Any]] = map(
            lambda t: {
                "year": t[0].year,
                "month": t[0].month,
                "variable": t[1],
                "overview_level": overview_level,
            },
            product(dates, variables),
        )
        uris: list[str] = list(
            self._resolve_wildcards(map(lambda fmt: uri_expanded.format(**fmt), fmts))
        )
        if not uris:
            exec_nodata_strat(
                f"resolver '{self.name}' found no files.", strategy=handle_nodata
            )
            return []  # if ignore

        return uris
