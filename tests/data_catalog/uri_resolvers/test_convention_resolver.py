from itertools import product
from string import Formatter
from typing import Any, Dict, Iterator

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from hydromt._typing import NoDataException
from hydromt.data_catalog.uri_resolvers.convention_resolver import ConventionResolver


class TestConventionResolver:
    @pytest.fixture(scope="class")
    def test_filesystem(self) -> MemoryFileSystem:
        template = "/{{unknown_key}}_{variable}_{year}_{month:02d}.nc"
        variables = ["precip", "temp"]
        years = ["2020", "2021"]
        months = range(1, 13)

        carthesian_prod: Iterator[str, str, str, str] = product(
            variables, years, months
        )
        formats: Iterator[Dict[str, Any]] = map(
            lambda _vars: dict(zip(("variable", "year", "month"), _vars)),
            carthesian_prod,
        )

        fs = MemoryFileSystem()
        for format in formats:
            uri = Formatter().format(template, **format)
            fs.touch(uri)
        return fs

    def test_resolves_correctly(self, test_filesystem: MemoryFileSystem):
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
                time_range=("2021-03-01", "2021-05-01"),
                variables=["precip"],
            )
        )
        assert len(uris) == 1 * 1 * 1 * 3
        assert uris[0] == "/{unknown_key}_precip_2021_03.nc"

    def test_raises_not_found(self, test_filesystem: MemoryFileSystem):
        uri = "/some_other_key/files/*"
        resolver = ConventionResolver(filesystem=test_filesystem)
        with pytest.raises(NoDataException):
            resolver.resolve(uri)

    def test_uri_without_wildcard(self, test_filesystem: MemoryFileSystem):
        uri = "/{unknown_key}_precip_2020_01.nc"
        resolver = ConventionResolver(filesystem=test_filesystem)
        assert resolver.resolve(uri) == [uri]

    def test_capture_regex(self):
        pat = "here-is-some-more-leading-{year}-text-for-{month}-you-{variable}.pq"
        example = "here-is-some-more-leading-2024-04-02-text-for-era5-you-0001.pq"
        resolver = ConventionResolver()

        glob, keys, regex = resolver._expand_uri_placeholders(pat)
        assert glob == "here-is-some-more-leading-*-text-for-*-you-*.pq"
        # we know regex will match, so type ignore is safe
        assert regex.match(example).groups() == ("2024-04-02", "era5", "0001")  # type: ignore
