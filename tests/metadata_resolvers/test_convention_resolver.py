import pytest
from polyfactory.factories.pydantic_factory import ModelFactory
from polyfactory.pytest_plugin import register_fixture

from hydromt.data_sources.data_source import DataSource
from hydromt.metadata_resolvers.convention_resolver import ConventionResolver


@register_fixture
class DataSourceFactory(ModelFactory[DataSource]):
    name = "test"
    data_type = "RasterDataset"
    uri = "/{unknown_key}_{zoom_level}_{variable}_{year}_{month:02d}.nc"
    metadata_resolver = ConventionResolver()


@pytest.mark.skip("see ticket #805 ")
class TestConventionResolver:
    def test_resolve(self, data_source_factory: DataSourceFactory):
        mock_source = data_source_factory.build()
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
