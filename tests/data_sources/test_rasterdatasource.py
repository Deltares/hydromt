from pathlib import Path

import pytest
import xarray as xr
from pydantic import ValidationError

from hydromt.data_sources import RasterDataSource
from hydromt.drivers.rasterdataset_driver import RasterDataSetDriver
from hydromt.metadata_resolvers import MetaDataResolver


class TestRasterDataSource:
    @pytest.fixture()
    def example_source(
        self,
        mock_rasterds_driver: RasterDataSetDriver,
        mock_resolver: MetaDataResolver,
        tmp_dir: Path,
    ):
        return RasterDataSource(
            root=".",
            name="example_rasterds",
            data_type="RasterDataSet",
            driver=mock_rasterds_driver,
            metadata_resolver=mock_resolver,
            uri=str(tmp_dir / "rasterds.zarr"),
        )

    def test_validators(self):
        with pytest.raises(ValidationError) as e_info:
            RasterDataSource(
                root=".",
                name="name",
                data_type="RasterDataSet",
                uri="uri",
                metadata_resolver="does not exist",
                driver="does not exist",
            )

        assert e_info.value.error_count() == 2
        error_meta = next(
            filter(lambda e: e["loc"] == ("metadata_resolver",), e_info.value.errors())
        )
        assert error_meta["type"] == "value_error"
        error_driver = next(
            filter(lambda e: e["loc"] == ("driver",), e_info.value.errors())
        )
        assert error_driver["type"] == "value_error"

    def test_model_validate(
        self, mock_rasterds_driver: RasterDataSetDriver, mock_resolver: MetaDataResolver
    ):
        RasterDataSource.model_validate(
            {
                "name": "zarrfile",
                "data_type": "RasterDataSet",
                "driver": mock_rasterds_driver,
                "metadata_resolver": mock_resolver,
                "uri": "test_uri",
            }
        )
        with pytest.raises(
            ValidationError, match="'data_type' must be 'RasterDataSet'."
        ):
            RasterDataSource.model_validate(
                {
                    "name": "geojsonfile",
                    "data_type": "DifferentDataType",
                    "driver": mock_rasterds_driver,
                    "metadata_resolver": mock_resolver,
                    "uri": "test_uri",
                }
            )

    def test_read_data(self, example_source: RasterDataSource, rasterds: xr.Dataset):
        assert rasterds == example_source.read_data()
