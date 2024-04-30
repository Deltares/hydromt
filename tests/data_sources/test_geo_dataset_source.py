from pathlib import Path
from typing import List, Type

import pytest
import xarray as xr
from pydantic import ValidationError

from hydromt._typing import StrPath
from hydromt.data_adapter import GeoDatasetAdapter
from hydromt.data_source import GeoDatasetSource, SourceMetadata
from hydromt.drivers import GeoDatasetDriver


@pytest.fixture()
def mock_raster_ds_adapter():
    class MockGeoDataSetAdapter(GeoDatasetSource):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, **kwargs):
            return ds

    return MockGeoDataSetAdapter()


class TestGeoDatasetSource:
    def test_validators(self, mock_raster_ds_adapter: GeoDatasetAdapter):
        with pytest.raises(ValidationError) as e_info:
            GeoDatasetSource(
                name="name",
                uri="uri",
                data_adapter=mock_raster_ds_adapter,
                driver="does not exist",
            )

        assert e_info.value.error_count() == 1
        error_driver = next(
            filter(lambda e: e["loc"] == ("driver",), e_info.value.errors())
        )
        assert error_driver["type"] == "model_type"

    def test_model_validate(
        self,
        mock_raster_ds_driver: GeoDatasetDriver,
        mock_raster_ds_adapter: GeoDatasetAdapter,
    ):
        GeoDatasetSource.model_validate(
            {
                "name": "zarrfile",
                "driver": mock_raster_ds_driver,
                "data_adapter": mock_raster_ds_adapter,
                "uri": "test_uri",
            }
        )
        with pytest.raises(ValidationError, match="'data_type' must be 'GeoDataset'."):
            GeoDatasetSource.model_validate(
                {
                    "name": "geojsonfile",
                    "data_type": "DifferentDataType",
                    "driver": mock_raster_ds_driver,
                    "data_adapter": mock_raster_ds_adapter,
                    "uri": "test_uri",
                }
            )

    def test_instantiate_directly(
        self,
    ):
        datasource = GeoDatasetSource(
            name="test",
            uri="points.zarr",
            zoom_levels={1: 10},
            driver={"name": "zarr", "metadata_resolver": "convention"},
            data_adapter={"unit_add": {"geoattr": 1.0}},
        )
        assert isinstance(datasource, GeoDatasetSource)

    def test_instantiate_directly_minimal_kwargs(self):
        GeoDatasetSource(
            name="test",
            uri="points.zarr",
            driver={"name": "zarr"},
        )

    def test_read_data(
        self,
        rasterds: xr.Dataset,
        mock_raster_ds_driver: GeoDatasetDriver,
        mock_raster_ds_adapter: GeoDatasetAdapter,
        tmp_dir: Path,
    ):
        source = GeoDatasetSource(
            root=".",
            name="example_rasterds",
            driver=mock_raster_ds_driver,
            data_adapter=mock_raster_ds_adapter,
            uri=str(tmp_dir / "rasterds.zarr"),
        )
        assert rasterds == source.read_data()

    @pytest.fixture()
    def MockDriver(self, rasterds: xr.Dataset):
        class MockGeoDatasetDriver(GeoDatasetDriver):
            name = "mock_geodf_to_file"

            def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
                pass

            def read(self, uri: str, **kwargs) -> xr.Dataset:
                return self.read_data([uri], **kwargs)

            def read_data(self, uris: List[str], **kwargs) -> xr.Dataset:
                return rasterds

        return MockGeoDatasetDriver

    def test_to_file(self, MockDriver: Type[GeoDatasetDriver]):
        mock_driver = MockDriver()

        source = GeoDatasetSource(
            name="test",
            uri="raster.nc",
            driver=mock_driver,
            metadata=SourceMetadata(crs=4326),
        )
        new_source = source.to_file("test")
        assert "local" in new_source.driver.filesystem.protocol
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(mock_driver) != id(new_source.driver)

    def test_to_file_override(self, MockDriver: Type[GeoDatasetDriver]):
        driver1 = MockDriver()
        source = GeoDatasetSource(
            name="test",
            uri="raster.nc",
            driver=driver1,
            metadata=SourceMetadata(crs=4326),
        )
        driver2 = MockDriver(filesystem="memory")
        new_source = source.to_file("test", driver_override=driver2)
        assert new_source.driver.filesystem.protocol == "memory"
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(driver2) == id(new_source.driver)
