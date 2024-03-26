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
