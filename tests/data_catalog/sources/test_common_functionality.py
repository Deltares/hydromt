from pathlib import Path
from typing import Type

import pytest
from pydantic import ValidationError

from hydromt._typing import SourceMetadata
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
from hydromt.data_catalog.uri_resolvers import URIResolver


class TestValidators:
    @pytest.mark.parametrize(
        ("source_cls", "_adapter"),
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
        ("source_cls", "_driver_cls", "_adapter"),
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
            driver={"name": driver_name, "uri_resolver": "convention"},
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

    @pytest.mark.parametrize(
        ("source_cls", "_driver_cls", "path"),
        [
            (DataFrameSource, "MockDataFrameDriver", "test.csv"),
            (DatasetSource, "MockDatasetDriver", "test.nc"),
            (GeoDataFrameSource, "MockGeoDataFrameDriver", "test.gpkg"),
            (GeoDatasetSource, "MockGeoDatasetDriver", "test.nc"),
            (RasterDatasetSource, "MockRasterDatasetDriver", "test.zarr"),
        ],
    )
    def test_to_file(
        self,
        source_cls: Type[DataSource],
        _driver_cls: str,  # noqa: PT019
        path: str,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ):
        driver_cls: Type[BaseDriver] = request.getfixturevalue(_driver_cls)
        driver = driver_cls()
        uri: str = str(tmp_path / path)
        source = source_cls(
            name="test",
            uri=uri,
            driver=driver,
            metadata=SourceMetadata(crs=4326),
        )
        new_source: DataSource = source.to_file(uri)
        assert "local" in new_source.driver.filesystem.protocol
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(driver) != id(new_source.driver)

    @pytest.mark.parametrize(
        ("source_cls", "_driver_cls", "_adapter", "path"),
        [
            (DataFrameSource, "MockDataFrameDriver", "mock_df_adapter", "test.csv"),
            (DatasetSource, "MockDatasetDriver", "mock_ds_adapter", "test.nc"),
            (
                GeoDataFrameSource,
                "MockGeoDataFrameDriver",
                "mock_geodataframe_adapter",
                "test.gpkg",
            ),
            (
                GeoDatasetSource,
                "MockGeoDatasetDriver",
                "mock_geo_ds_adapter",
                "test.nc",
            ),
            (
                RasterDatasetSource,
                "MockRasterDatasetDriver",
                "mock_raster_ds_adapter",
                "test.zarr",
            ),
        ],
    )
    def test_to_file_override(
        self,
        source_cls: Type[DataSource],
        _driver_cls: str,  # noqa: PT019
        _adapter: str,  # noqa: PT019
        path: str,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ):
        driver_cls: Type[BaseDriver] = request.getfixturevalue(_driver_cls)
        adapter: Type[DataAdapterBase] = request.getfixturevalue(_adapter)
        driver1 = driver_cls()

        uri: str = str(tmp_path / path)
        source: DataSource = source_cls(
            name="test", uri=uri, driver=driver1, data_adapter=adapter
        )
        driver2 = driver_cls(filesystem="memory")
        new_source = source.to_file("test", driver_override=driver2)
        assert new_source.driver.filesystem.protocol == "memory"
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(driver2) == id(new_source.driver)

    @pytest.mark.parametrize(
        ("source_cls", "_driver_cls", "_adapter", "path"),
        [
            (
                DataFrameSource,
                "MockDataFrameReadOnlyDriver",
                "mock_df_adapter",
                "test.csv",
            ),
            (DatasetSource, "MockDatasetReadOnlyDriver", "mock_ds_adapter", "test.nc"),
            (
                GeoDataFrameSource,
                "MockGeoDataFrameReadOnlyDriver",
                "mock_geodataframe_adapter",
                "test.gpkg",
            ),
            (
                GeoDatasetSource,
                "MockGeoDatasetReadOnlyDriver",
                "mock_geo_ds_adapter",
                "test.nc",
            ),
            (
                RasterDatasetSource,
                "MockRasterDatasetReadOnlyDriver",
                "mock_raster_ds_adapter",
                "test.zarr",
            ),
        ],
    )
    def test_to_file_defaults(
        self,
        source_cls: Type[DataSource],
        _driver_cls: str,  # noqa: PT019
        _adapter: str,  # noqa: PT019
        path: str,
        tmp_path: Path,
        mock_resolver: URIResolver,
        request: pytest.FixtureRequest,
    ):
        driver_cls: Type[BaseDriver] = request.getfixturevalue(_driver_cls)
        adapter: Type[DataAdapterBase] = request.getfixturevalue(_adapter)

        old_path: Path = tmp_path / path
        new_path: Path = tmp_path / path.replace("test", "temp")

        new_source: DataSource = source_cls(
            name="test",
            uri=str(old_path),
            data_adapter=adapter,
            driver=driver_cls(uri_resolver=mock_resolver),
        ).to_file(new_path)
        # assert new_path.is_file()
        assert new_source.root is None
        assert new_source.driver.filesystem.protocol == ("file", "local")

    @pytest.mark.parametrize(
        ("source_cls", "_driver_cls"),
        [
            (DataFrameSource, "MockDataFrameReadOnlyDriver"),
            (DatasetSource, "MockDatasetReadOnlyDriver"),
            (GeoDataFrameSource, "MockGeoDataFrameReadOnlyDriver"),
            (GeoDatasetSource, "MockGeoDatasetReadOnlyDriver"),
            (RasterDatasetSource, "MockRasterDatasetReadOnlyDriver"),
        ],
    )
    def test_raises_on_write_with_incapable_driver(
        self,
        source_cls: Type[DataSource],
        _driver_cls: str,  # noqa: PT019
        request: pytest.FixtureRequest,
    ):
        driver_cls: Type[BaseDriver] = request.getfixturevalue(_driver_cls)
        mock_driver = driver_cls()
        source = source_cls(
            name="test",
            uri="test",
            driver=mock_driver,
        )
        with pytest.raises(
            RuntimeError,
            match=f"driver: '{mock_driver.name}' does not support writing data",
        ):
            source.to_file("/non/existant/test", driver_override=mock_driver)
