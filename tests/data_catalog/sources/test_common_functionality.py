from typing import Type

import pytest
from pydantic import ValidationError

from hydromt.data_catalog.adapters.data_adapter_base import DataAdapterBase
from hydromt.data_catalog.drivers import BaseDriver
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

    @pytest.mark.parametrize(
        "source_cls,_driver_cls,_adapter",  # noqa: PT006
        [
            (DataFrameSource, "MockDataFrameDriver", "mock_df_adapter"),
            (DatasetSource, "MockDatasetDriver", "mock_ds_adapter"),
            (GeoDataFrameSource, "MockGeoDataFrameDriver", "mock_geodataframe_adapter"),
            (GeoDatasetSource, "MockGeoDatasetDriver", "mock_geo_ds_adapter"),
            (RasterDatasetSource, "MockRasterDatasetDriver", "mock_raster_ds_adapter"),
        ],
    )
    def test_model_validates_correctly(
        self,
        source_cls: Type[DataSource],
        _driver_cls: str,  # noqa: PT019
        _adapter: str,  # noqa: PT019
        request: pytest.FixtureRequest,
    ):
        driver_cls: BaseDriver = request.getfixturevalue(_driver_cls)
        adapter: DataAdapterBase = request.getfixturevalue(_adapter)

        source_cls.model_validate(
            {
                "name": "example_file",
                "driver": driver_cls(),
                "data_adapter": adapter,
                "uri": "test_uri",
            }
        )
        with pytest.raises(ValidationError, match="'data_type' must be 'DataFrame'."):
            DataFrameSource.model_validate(
                {
                    "name": "geojsonfile",
                    "data_type": "DifferentDataType",
                    "driver": driver_cls(),
                    "data_adapter": adapter,
                    "uri": "test_uri",
                }
            )

    @pytest.mark.parametrize(
        "source_cls,driver_name",  # noqa: PT006
        [
            (DataFrameSource, "pandas"),
            (DatasetSource, "dataset_xarray"),
            (GeoDataFrameSource, "pyogrio"),
            (GeoDatasetSource, "geodataset_xarray"),
            (RasterDatasetSource, "raster_xarray"),
        ],
    )
    def test_instantiate_directly(
        self,
        source_cls: Type[DataSource],
        driver_name: str,
    ):
        datasource = source_cls(
            name="test",
            uri="data.format",
            driver={"name": driver_name, "metadata_resolver": "convention"},
            data_adapter={"unit_add": {"geoattr": 1.0}},
        )
        assert isinstance(datasource, source_cls)

    @pytest.mark.parametrize(
        "source_cls,driver_name",  # noqa: PT006
        [
            (DataFrameSource, "pandas"),
            (DatasetSource, "dataset_xarray"),
            (GeoDataFrameSource, "pyogrio"),
            (GeoDatasetSource, "geodataset_xarray"),
            (RasterDatasetSource, "raster_xarray"),
        ],
    )
    def test_instantiate_directly_minimal_kwargs(
        self, source_cls: Type[DataSource], driver_name: str
    ):
        source_cls(
            name="test",
            uri="points.csv",
            driver=driver_name,
        )
