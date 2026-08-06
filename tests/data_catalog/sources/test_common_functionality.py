from pathlib import Path
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
from hydromt.data_catalog.uri_resolvers import ConventionResolver, URIResolver
from hydromt.error import NoDataException, NoDataStrategy
from hydromt.typing import SourceMetadata


class TestValidators:
    def _source(
        self,
        request: pytest.FixtureRequest,
        source_cls: Type[DataSource],
        driver_fixture: str,
        adapter_fixture: str,
        uri: str,
    ) -> DataSource:
        return source_cls(
            name="test",
            uri=uri,
            driver=request.getfixturevalue(driver_fixture)(),
            data_adapter=request.getfixturevalue(adapter_fixture),
            uri_resolver=ConventionResolver(),
            metadata=SourceMetadata(crs=4326),
        )

    @pytest.mark.parametrize(
        ("source_cls", "_driver_cls", "_adapter", "fixture_year"),
        [
            (DatasetSource, "MockDatasetDriver", "mock_ds_adapter", 2020),
            (GeoDatasetSource, "MockGeoDatasetDriver", "mock_geo_ds_adapter", 2000),
            (
                RasterDatasetSource,
                "MockRasterDatasetDriver",
                "mock_raster_ds_adapter",
                2014,
            ),
        ],
    )
    def test_read_data_checks_years_from_resolved_multi_file_uris(
        self,
        source_cls: Type[DataSource],
        _driver_cls: str,  # noqa: PT019
        _adapter: str,  # noqa: PT019
        fixture_year: int,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ):
        # Mock drivers ignore URIs and return fixed fixture data for `fixture_year`;
        # touched files only make ConventionResolver expect extra years.
        missing = list(range(fixture_year + 1, fixture_year + 4))
        for year in [fixture_year, *missing]:
            (tmp_path / f"data_{year}.nc").touch()
        source = self._source(
            request, source_cls, _driver_cls, _adapter, str(tmp_path / "data_{year}.nc")
        )

        with pytest.raises(NoDataException) as exc:
            source.read_data()
        assert "missing years" in str(exc.value)
        assert all(str(year) in str(exc.value) for year in missing)

    def test_read_data_checks_month_only_placeholders_from_resolved_multi_file_uris(
        self,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ):
        for month in ["09", "10"]:
            (tmp_path / f"data_{month}.nc").touch()
        source = self._source(
            request,
            RasterDatasetSource,
            "MockRasterDatasetDriver",
            "mock_raster_ds_adapter",
            str(tmp_path / "data_{month:02d}.nc"),
        )

        with pytest.raises(NoDataException, match=r"missing months.*10"):
            source.read_data()

    def test_read_data_checks_year_month_placeholders_across_year_boundary(
        self,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ):
        months = [(2020, month) for month in range(1, 13)] + [(2021, 1), (2021, 2)]
        for year, month in months:
            (tmp_path / f"data_{year}_{month:02d}.nc").touch()
        source = self._source(
            request,
            DatasetSource,
            "MockDatasetDriver",
            "mock_ds_adapter",
            str(tmp_path / "data_{year}_{month:02d}.nc"),
        )

        with pytest.raises(NoDataException) as exc:
            source.read_data()
        assert "missing months" in str(exc.value)
        assert "2021-01" in str(exc.value)
        assert "2021-02" in str(exc.value)
        assert "missing years" not in str(exc.value)

    def test_read_data_warns_placeholder_mismatch_without_dropping_data(
        self,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ):
        for year in [2020, 2021]:
            (tmp_path / f"data_{year}.nc").touch()
        source = self._source(
            request,
            DatasetSource,
            "MockDatasetDriver",
            "mock_ds_adapter",
            str(tmp_path / "data_{year}.nc"),
        )

        assert source.read_data(handle_nodata=NoDataStrategy.WARN) is not None

    def test_read_data_skips_placeholder_check_for_single_uri(
        self,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ):
        (tmp_path / "data_2021.nc").touch()
        source = self._source(
            request,
            DatasetSource,
            "MockDatasetDriver",
            "mock_ds_adapter",
            str(tmp_path / "data_{year}.nc"),
        )

        source.read_data()

    @pytest.mark.parametrize(
        ("source_cls", "_adapter"),
        [
            (DataFrameSource, "mock_df_adapter"),
            (DatasetSource, "mock_ds_adapter"),
            (GeoDataFrameSource, "mock_gdf_adapter"),
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
            (GeoDataFrameSource, "MockGeoDataFrameDriver", "mock_gdf_adapter"),
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
            driver={"name": driver_name},
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
        mock_resolver: URIResolver,
        request: pytest.FixtureRequest,
    ):
        driver_cls: Type[BaseDriver] = request.getfixturevalue(_driver_cls)
        driver = driver_cls()
        uri: str = str(tmp_path / path)
        source = source_cls(
            name="test",
            uri=uri,
            uri_resolver=mock_resolver,
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
                "mock_gdf_adapter",
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
        mock_resolver: URIResolver,
        request: pytest.FixtureRequest,
    ):
        driver_cls: Type[BaseDriver] = request.getfixturevalue(_driver_cls)
        adapter: Type[DataAdapterBase] = request.getfixturevalue(_adapter)
        driver1 = driver_cls()

        uri: str = str(tmp_path / path)
        source: DataSource = source_cls(
            name="test",
            uri=uri,
            uri_resolver=mock_resolver,
            driver=driver1,
            data_adapter=adapter,
        )
        driver2 = driver_cls(filesystem="memory")
        new_source = source.to_file("test", driver_override=driver2)
        assert new_source.driver.filesystem.get_fs().protocol == "memory"
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
                "mock_gdf_adapter",
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
            uri_resolver=mock_resolver,
            driver=driver_cls(),
        ).to_file(new_path)
        assert new_source.root is None
        assert new_source.driver.filesystem.protocol == "file"
        assert new_source.driver.filesystem.get_fs().protocol == ("file", "local")

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
