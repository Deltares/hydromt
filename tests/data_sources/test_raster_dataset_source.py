from pathlib import Path

import pytest
import xarray as xr
from pydantic import ValidationError

from hydromt.data_adapter import RasterDatasetAdapter
from hydromt.data_source import RasterDatasetSource
from hydromt.driver.rasterdataset_driver import RasterDatasetDriver


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
        assert error_driver["type"] == "modeltype"

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
