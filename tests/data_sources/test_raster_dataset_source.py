from pathlib import Path
from typing import Type

import pytest
import xarray as xr
from pydantic import ValidationError

from hydromt._typing import StrPath
from hydromt.data_adapter import RasterDatasetAdapter
from hydromt.data_source import RasterDatasetSource
from hydromt.drivers.rasterdataset_driver import RasterDatasetDriver


@pytest.fixture()
def mock_raster_ds_adapter():
    class MockRasterDataSetAdapter(RasterDatasetAdapter):
        def transform(self, ds: xr.Dataset, **kwargs):
            return ds

    return MockRasterDataSetAdapter()


class TestRasterDatasetSource:
    def test_validators(self, mock_raster_ds_adapter: RasterDatasetAdapter):
        with pytest.raises(ValidationError) as e_info:
            RasterDatasetSource(
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
        mock_raster_ds_driver: RasterDatasetDriver,
        mock_raster_ds_adapter: RasterDatasetAdapter,
    ):
        RasterDatasetSource.model_validate(
            {
                "name": "zarrfile",
                "driver": mock_raster_ds_driver,
                "data_adapter": mock_raster_ds_adapter,
                "uri": "test_uri",
            }
        )
        with pytest.raises(
            ValidationError, match="'data_type' must be 'RasterDataset'."
        ):
            RasterDatasetSource.model_validate(
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
        datasource = RasterDatasetSource(
            name="test",
            uri="points.zarr",
            zoom_levels={1: 10},
            driver={"name": "zarr", "metadata_resolver": "convention"},
            data_adapter={"unit_add": {"geoattr": 1.0}},
        )
        assert isinstance(datasource, RasterDatasetSource)

    def test_instantiate_directly_minimal_kwargs(self):
        RasterDatasetSource(
            name="test",
            uri="points.zarr",
            driver={"name": "zarr"},
        )

    def test_read_data(
        self,
        rasterds: xr.Dataset,
        mock_raster_ds_driver: RasterDatasetDriver,
        mock_raster_ds_adapter: RasterDatasetAdapter,
        tmp_dir: Path,
    ):
        source = RasterDatasetSource(
            root=".",
            name="example_rasterds",
            driver=mock_raster_ds_driver,
            data_adapter=mock_raster_ds_adapter,
            uri=str(tmp_dir / "rasterds.zarr"),
        )
        assert rasterds == source.read_data()

    @pytest.fixture()
    def MockDriver(self, rasterds: xr.Dataset):
        class MockRasterDatasetDriver(RasterDatasetDriver):
            name = "mock_geodf_to_file"

            def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
                pass

            def read(self, uri: str, **kwargs) -> xr.Dataset:
                return rasterds

        return MockRasterDatasetDriver

    def test_to_file(self, MockDriver: Type[RasterDatasetDriver]):
        mock_driver = MockDriver()

        source = RasterDatasetSource(
            name="test", uri="raster.nc", driver=mock_driver, crs=4326
        )
        new_source = source.to_file("test")
        assert "local" in new_source.driver.filesystem.protocol
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(mock_driver) != id(new_source.driver)

    def test_to_file_override(self, MockDriver: Type[RasterDatasetDriver]):
        driver1 = MockDriver()
        source = RasterDatasetSource(
            name="test", uri="raster.nc", driver=driver1, crs=4326
        )
        driver2 = MockDriver(filesystem="memory")
        new_source = source.to_file("test", driver_override=driver2)
        assert new_source.driver.filesystem.protocol == "memory"
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(driver2) == id(new_source.driver)