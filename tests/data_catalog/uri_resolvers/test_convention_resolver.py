import logging
import os
from itertools import product
from string import Formatter
from typing import Any, Dict, Iterator

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from hydromt._utils.naming_convention import _expand_uri_placeholders
from hydromt.data_catalog.uri_resolvers.convention_resolver import (
    ConventionResolver,
    _normalize_local_path,
)
from hydromt.error import NoDataException
from hydromt.typing.fsspec_types import FSSpecFileSystem
from hydromt.typing.type_def import TimeRange


class TestConventionResolver:
    @pytest.fixture(scope="class")
    def test_filesystem(self) -> FSSpecFileSystem:
        template = "/{{unknown_key}}_{variable}_{year}_{month:02d}.nc"
        variables = ["precip", "temp"]
        years = ["2020", "2021"]
        months = range(1, 13)

        carthesian_prod: Iterator[str, str, str, str] = product(
            variables, years, months
        )
        formats: Iterator[Dict[str, Any]] = map(
            lambda _vars: dict(zip(("variable", "year", "month"), _vars, strict=False)),
            carthesian_prod,
        )

        fs = MemoryFileSystem()
        for format in formats:
            uri = Formatter().format(template, **format)
            fs.touch(uri)

        file_system = FSSpecFileSystem()
        file_system._fs = fs
        return file_system

    def test_resolves_correctly(self, test_filesystem: FSSpecFileSystem):
        uri = "/{unknown_key}_{variable}_{year}_{month:02d}.nc"
        resolver = ConventionResolver(filesystem=test_filesystem)

        uris = sorted(resolver.resolve(uri))  # sort for assertion
        assert len(uris) == 2 * 2 * 12  # zoom levels * variables * years * months
        assert uris[0] == "/{unknown_key}_precip_2020_01.nc"

        uris = sorted(resolver.resolve(uri, variables=["temp"]))
        assert len(uris) == 1 * 2 * 12
        assert uris[0] == "/{unknown_key}_temp_2020_01.nc"

        uris = sorted(
            resolver.resolve(
                uri,
                time_range=TimeRange(start="2021-03-01", end="2021-05-01"),
                variables=["precip"],
            )
        )
        assert len(uris) == 1 * 1 * 1 * 3
        assert uris[0] == "/{unknown_key}_precip_2021_03.nc"

    def test_raises_not_found(self, test_filesystem: FSSpecFileSystem):
        uri = "/some_other_key/files/*"
        resolver = ConventionResolver(filesystem=test_filesystem)
        with pytest.raises(NoDataException):
            resolver.resolve(uri)

    def test_uri_without_wildcard(self, test_filesystem: FSSpecFileSystem):
        uri = "/{unknown_key}_precip_2020_01.nc"
        resolver = ConventionResolver(filesystem=test_filesystem)
        assert resolver.resolve(uri) == [uri]

    def test_resolver_logs_all_expanded_uris(
        self, test_filesystem: FSSpecFileSystem, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.DEBUG, logger="hydromt")
        resolver = ConventionResolver(filesystem=test_filesystem)
        uri = "/{unknown_key}_{variable}_{year}_{month:02d}.nc"
        resolved_uris = resolver.resolve(uri)

        assert (
            "Resolver 'convention' found 48 files at /{unknown_key}_*_*_*.nc:"
            in caplog.text
        )
        for resolved_uri in resolved_uris:
            assert resolved_uri in caplog.text

    def test_resolver_normalizes_windows_paths(self, monkeypatch: pytest.MonkeyPatch):

        # Patch wildcards to fake a Windows filesystem
        def _resolve_wildcards(
            self,
            uris: Iterator[str],  # noqa: ARG001
        ) -> list[str]:
            return [
                "C:/data/data_2020_01.nc",
                "C:/data/data_2020_02.nc",
            ]

        monkeypatch.setattr(
            ConventionResolver, "_resolve_wildcards", _resolve_wildcards
        )
        resolver = ConventionResolver()

        monkeypatch.setattr(os, "name", "nt")
        resolver.resolve(r"C:\data\data_{year}_{month:02d}.nc")
        assert resolver._resolved_uri_placeholders == [
            {"year": "2020", "month": "01"},
            {"year": "2020", "month": "02"},
        ]

        monkeypatch.setattr(os, "name", "posix")
        assert _normalize_local_path(r"/data/data\2020.nc") == r"/data/data\2020.nc"

    def test_capture_regex(self):
        pat = "here-is-some-more-leading-{year}-text-for-{month}-you-{variable}.pq"
        example = "here-is-some-more-leading-2024-04-02-text-for-era5-you-0001.pq"

        glob, _, regex = _expand_uri_placeholders(
            pat, placeholders=["year", "month", "variable"]
        )
        assert glob == "here-is-some-more-leading-*-text-for-*-you-*.pq"
        # we know regex will match, so type ignore is safe
        assert regex.match(example).groups() == ("2024-04-02", "era5", "0001")  # type: ignore
