from pathlib import Path
from typing import ClassVar, List, Type

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from hydromt._typing import SourceMetadata, StrPath
from hydromt.data_catalog.adapters import RasterDatasetAdapter
from hydromt.data_catalog.drivers import RasterDatasetDriver
from hydromt.data_catalog.sources import RasterDatasetSource
from hydromt.gis.utils import to_geographic_bbox


@pytest.fixture()
def mock_raster_ds_adapter():
    class MockRasterDataSetAdapter(RasterDatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, **kwargs):
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
        assert error_driver["type"] == "value_error"

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
            driver={"name": "raster_xarray", "metadata_resolver": "convention"},
            data_adapter={"unit_add": {"geoattr": 1.0}},
        )
        assert isinstance(datasource, RasterDatasetSource)

    def test_instantiate_directly_minimal_kwargs(self):
        RasterDatasetSource(
            name="test",
            uri="points.zarr",
            driver={"name": "raster_xarray"},
        )

    def test_read_data(
        self,
        raster_ds: xr.Dataset,
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
        assert raster_ds == source.read_data()

    @pytest.fixture()
    def MockDriver(self, raster_ds: xr.Dataset):
        class MockRasterDatasetDriver(RasterDatasetDriver):
            name = "mock_rasterds_to_file"

            def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
                pass

            def read(self, uri: str, **kwargs) -> xr.Dataset:
                return self.read_data([uri], **kwargs)

            def read_data(self, uris: List[str], **kwargs) -> xr.Dataset:
                return raster_ds

        return MockRasterDatasetDriver

    @pytest.fixture()
    def MockWritableDriver(self, raster_ds: xr.Dataset):
        class MockWritableRasterDatasetDriver(RasterDatasetDriver):
            name = "mock_rasterds_to_file"
            supports_writing: ClassVar[bool] = True

            def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
                pass

            def read(self, uri: str, **kwargs) -> xr.Dataset:
                return self.read_data([uri], **kwargs)

            def read_data(self, uris: List[str], **kwargs) -> xr.Dataset:
                return raster_ds

        return MockWritableRasterDatasetDriver

    @pytest.fixture()
    def writable_source(
        self, MockWritableDriver: Type[RasterDatasetDriver]
    ) -> RasterDatasetSource:
        return RasterDatasetSource(
            name="test",
            uri="raster.zarr",
            driver=MockWritableDriver(),
            metadata=SourceMetadata(crs=4326),
        )

    def test_to_file(self, writable_source: RasterDatasetSource):
        new_source = writable_source.to_file("test")
        assert "local" in new_source.driver.filesystem.protocol
        # make sure we are not changing the state
        assert id(new_source) != id(writable_source)
        assert id(writable_source.driver) != id(new_source.driver)

    def test_to_file_override(
        self,
        writable_source: RasterDatasetSource,
        MockWritableDriver: Type[RasterDatasetDriver],
    ):
        driver = MockWritableDriver(filesystem="memory")
        new_source = writable_source.to_file("test", driver_override=driver)
        assert new_source.driver.filesystem.protocol == "memory"
        # make sure we are not changing the state
        assert id(new_source) != id(writable_source)
        assert id(driver) == id(new_source.driver)

    def test_detect_extent(
        self, writable_source: RasterDatasetSource, rioda: xr.DataArray
    ):
        rioda_expected_bbox = (3.0, -11.0, 6.0, -9.0)
        rioda_detected_bbox = to_geographic_bbox(*writable_source.detect_bbox(rioda))

        assert np.all(np.equal(rioda_expected_bbox, rioda_detected_bbox))
