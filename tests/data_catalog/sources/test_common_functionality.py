from typing import Type

import pytest
from pydantic import ValidationError

from hydromt.data_catalog.adapters.data_adapter_base import DataAdapterBase
from hydromt.data_catalog.sources import (
    DataFrameSource,
    DatasetSource,
    DataSource,
    GeoDataFrameSource,
    GeoDatasetSource,
    RasterDatasetSource,
)


class TestValidators:
    @pytest.mark.parametrize(
        "source_cls,_adapter",  # noqa: PT006
        [
            (DataFrameSource, "mock_df_adapter"),
            (DatasetSource, "mock_ds_adapter"),
            (GeoDataFrameSource, "mock_geodataframe_adapter"),
            (GeoDatasetSource, "mock_geo_ds_adapter"),
            (RasterDatasetSource, "mock_raster_ds_adapter"),
        ],
    )
    def test_validators(
        self,
        source_cls: Type[DataSource],
        _adapter: str,  # noqa: PT019
        request: pytest.FixtureRequest,
    ):
        adapter: DataAdapterBase = request.getfixturevalue(_adapter)
        with pytest.raises(ValidationError) as e_info:
            source_cls(
                name="name",
                uri="uri",
                data_adapter=adapter,
                driver="does_not_exist",
            )

        assert e_info.value.error_count() == 1
        error_driver = next(
            filter(lambda e: e["loc"] == ("driver",), e_info.value.errors())
        )
        assert error_driver["type"] == "value_error"
