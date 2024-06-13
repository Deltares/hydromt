from typing import Any, Dict, Tuple, Type

import pytest
from pydantic import ValidationError

from hydromt.data_catalog.adapters import DataFrameAdapter, DatasetAdapter
from hydromt.data_catalog.sources import (
    DataFrameSource,
    DatasetSource,
    GeoDataFrameSource,
    GeoDatasetSource,
    RasterDatasetSource,
)


class TestValidators:
    @pytest.fixture()
    def dataframe_source_args(
        self, mock_df_adapter: DataFrameAdapter
    ) -> Tuple[Type[DataFrameSource], Dict[str, Any]]:
        return (
            DataFrameSource,
            dict(
                name="name",
                uri="uri",
                data_adapter=mock_df_adapter,
                driver="does not exist",
            ),
        )

    @pytest.fixture()
    def dataset_source_args(
        self, mock_ds_adapter: DatasetAdapter
    ) -> Tuple[Type[DatasetSource], Dict[str, Any]]:
        return (
            DatasetSource,
            dict(
                name="name",
                uri="uri",
                data_adapter=mock_ds_adapter,
                driver="does not exist",
            ),
        )

    @pytest.fixture()
    def geodataframe_source_args(
        self, mock_geodataframe_adapter
    ) -> Tuple[Type[GeoDataFrameSource], Dict[str, Any]]:
        return (
            GeoDataFrameSource,
            dict(
                name="name",
                uri="uri",
                data_adapter=mock_geodataframe_adapter,
                driver="does not exist",
            ),
        )

    @pytest.fixture()
    def geodataset_source_args(
        self, mock_geo_ds_adapter
    ) -> Tuple[Type[GeoDatasetSource], Dict[str, Any]]:
        return (
            GeoDatasetSource,
            dict(
                name="name",
                uri="uri",
                data_adapter=mock_geo_ds_adapter,
                driver="does not exist",
            ),
        )

    @pytest.fixture()
    def rasterdataset_source_args(
        self, mock_raster_ds_adapter
    ) -> Tuple[Type[RasterDatasetSource], Dict[str, Any]]:
        return (
            RasterDatasetSource,
            dict(
                name="name",
                uri="uri",
                data_adapter=mock_raster_ds_adapter,
                driver="does not exist",
            ),
        )

    fixture_sources = pytest.mark.parametrize(
        "source",
        [
            "dataframe_source_args",
            "dataset_source_args",
            "geodataframe_source_args",
            "geodataset_source_args",
            "rasterdataset_source_args",
        ],
    )

    @fixture_sources
    def test_validators(self, source: str, request: pytest.FixtureRequest):
        source_cls, source_vals = request.getfixturevalue(source)
        with pytest.raises(ValidationError) as e_info:
            source_cls(**source_vals)

        assert e_info.value.error_count() == 1
        error_driver = next(
            filter(lambda e: e["loc"] == ("driver",), e_info.value.errors())
        )
        assert error_driver["type"] == "value_error"
