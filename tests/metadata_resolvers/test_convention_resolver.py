from itertools import product
from string import Formatter
from typing import Any, Dict, Iterator

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from hydromt.metadata_resolver.convention_resolver import ConventionResolver


class TestConventionResolver:
    @pytest.fixture(scope="class")
    def test_filesystem(self) -> MemoryFileSystem:
        template = "/{{unknown_key}}_{zoom_level}_{variable}_{year}_{month:02d}.nc"
        zoom_levels = ["0", "1"]
        variables = ["precip", "temp"]
        years = ["2020", "2021"]
        months = range(1, 13)

        carthesian_prod: Iterator[str, str, str, str] = product(
            zoom_levels, variables, years, months
        )
        formats: Iterator[Dict[str, Any]] = map(
            lambda _vars: dict(zip(("zoom_level", "variable", "year", "month"), _vars)),
            carthesian_prod,
        )

        fs = MemoryFileSystem()
        for format in formats:
            uri = Formatter().format(template, **format)
            fs.touch(uri)
        return fs

    def test_resolves_correctly(self, test_filesystem: MemoryFileSystem):
        uri = "/{unknown_key}_{zoom_level}_{variable}_{year}_{month:02d}.nc"
        resolver = ConventionResolver()

        uris = sorted(resolver.resolve(uri, test_filesystem))  # sort for assertion
        assert len(uris) == 1 * 2 * 2 * 12  # zoom levels * variables * years * months
        assert uris[0] == "/{unknown_key}_0_precip_2020_01.nc"

        uris = sorted(resolver.resolve(uri, test_filesystem, variables=["temp"]))
        assert len(uris) == 1 * 1 * 2 * 12
        assert uris[0] == "/{unknown_key}_0_temp_2020_01.nc"

        uris = sorted(
            resolver.resolve(
                uri,
                test_filesystem,
                timerange=("2021-03-01", "2021-05-01"),
                variables=["precip"],
            )
        )
        assert len(uris) == 1 * 1 * 1 * 3
        assert uris[0] == "/{unknown_key}_0_precip_2021_03.nc"

    def test_raises_not_found(self, test_filesystem: MemoryFileSystem):
        uri = "/some_other_key/files/*"
        resolver = ConventionResolver()
        with pytest.raises(FileNotFoundError):
            resolver.resolve(uri, test_filesystem)

    def test_uri_without_wildcard(self, test_filesystem: MemoryFileSystem):
        uri = "/{unknown_key}_0_precip_2020_01.nc"
        resolver = ConventionResolver()
        assert resolver.resolve(uri, test_filesystem) == [uri]

    def test_capture_regex(self):
        pat = "here-is-some-more-leading-{year}-text-for-{month}-you-{variable}.pq"
        example = "here-is-some-more-leading-2024-04-02-text-for-era5-you-0001.pq"
        resolver = ConventionResolver()

        glob, keys, regex = resolver._expand_uri_placeholders(pat)
        assert glob == "here-is-some-more-leading-*-text-for-*-you-*.pq"
        # we know regex will match, so type ignore is safe
        assert regex.match(example).groups() == ("2024-04-02", "era5", "0001")  # type: ignore
