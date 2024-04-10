from hydromt.metadata_resolver.convention_resolver import ConventionResolver


class TestConventionResolver:
    def test_resolve(self):
        uri = "/{unknown_key}_{zoom_level}_{variable}_{year}_{month:02d}.nc"
        resolver = ConventionResolver()
        # test
        uris = resolver.resolve(uri)
        assert len(uris) == 1
        assert uris[0] == "/{unknown_key}_0_*_*_*.nc"

        uris = resolver.resolve(uri, variables=["precip"])
        assert len(uris) == 1
        assert uris[0] == "/{unknown_key}_0_precip_*_*.nc"

        uris = resolver.resolve(
            uri, timerange=("2021-03-01", "2021-05-01"), variables=["precip"]
        )
        assert len(uris) == 3
        assert uris[0] == "/{unknown_key}_0_precip_2021_03.nc"

    def test_capture_regex(self):
        pat = "here-is-some-more-leading-{year}-text-for-{month}-you-{variable}.pq"
        example = "here-is-some-more-leading-2024-04-02-text-for-era5-you-0001.pq"
        resolver = ConventionResolver()

        glob, keys, regex = resolver._expand_uri_placeholders(pat)
        assert glob == "here-is-some-more-leading-*-text-for-*-you-*.pq"
        # we know regex will match, so type ignore is safe
        assert regex.match(example).groups() == ("2024-04-02", "era5", "0001")  # type: ignore
