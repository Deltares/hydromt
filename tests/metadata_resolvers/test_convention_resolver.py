import pytest

from hydromt.data_sources.rasterdataset import RasterDataSource
from hydromt.drivers.zarr_driver import ZarrDriver
from hydromt.metadata_resolvers.convention_resolver import ConventionResolver


@pytest.fixture()
def mock_source() -> RasterDataSource:
    return RasterDataSource(
        name="test",
        uri="/{unknown_key}_{zoom_level}_{variable}_{year}_{month:02d}.nc",
        metadata_resolver=ConventionResolver(),
        driver=ZarrDriver(),
    )


class TestConventionResolver:
    def test_resolve(self, mock_source: RasterDataSource):
        resolver = ConventionResolver()
        # test
        uris = resolver.resolve(mock_source)
        assert len(uris) == 1
        assert uris[0] == "/{unknown_key}_0_*_*_*.nc"

        uris = resolver.resolve(mock_source, variables=["precip"])
        assert len(uris) == 1
        assert uris[0] == "/{unknown_key}_0_precip_*_*.nc"

        uris = resolver.resolve(
            mock_source, timerange=("2021-03-01", "2021-05-01"), variables=["precip"]
        )
        assert len(uris) == 3
        assert uris[0] == "/{unknown_key}_0_precip_2021_03.nc"
