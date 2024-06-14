from pathlib import Path
from typing import ClassVar, List, Type

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydromt._typing import SourceMetadata, StrPath
from hydromt.data_catalog.adapters import GeoDatasetAdapter
from hydromt.data_catalog.drivers import GeoDatasetDriver, GeoDatasetXarrayDriver
from hydromt.data_catalog.sources import GeoDatasetSource
from hydromt.data_catalog.uri_resolvers.metadata_resolver import MetaDataResolver
from hydromt.gis.gis_utils import to_geographic_bbox


class TestGeoDatasetSource:
    def test_instantiate_directly(
        self,
    ):
        datasource = GeoDatasetSource(
            name="test",
            uri="points.zarr",
            driver={"name": "geodataset_vector", "metadata_resolver": "convention"},
            data_adapter={"unit_add": {"geoattr": 1.0}},
        )
        assert isinstance(datasource, GeoDatasetSource)

    def test_instantiate_directly_minimal_kwargs(self):
        GeoDatasetSource(
            name="test",
            uri="points.zarr",
            driver={"name": "geodataset_vector"},
        )

    def test_read_data(
        self,
        geoda: xr.DataArray,
        mock_geo_ds_driver: GeoDatasetDriver,
        mock_geo_ds_adapter: GeoDatasetAdapter,
        tmp_dir: Path,
    ):
        geoda = geoda.to_dataset()
        source = GeoDatasetSource(
            root=".",
            name="geoda.zarr",
            driver=mock_geo_ds_driver,
            data_adapter=mock_geo_ds_adapter,
            uri=str(tmp_dir / "geoda.zarr"),
        )
        read_data = source.read_data()
        assert read_data.equals(geoda)

    @pytest.fixture()
    def MockWritableDriver(self, geoda: xr.Dataset):
        class MockWriteableGeoDatasetDriver(GeoDatasetDriver):
            name = "mock_geods_to_file"
            supports_writing: ClassVar[bool] = True

            def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
                pass

            def read(self, uri: str, metadata: SourceMetadata, **kwargs) -> xr.Dataset:
                kinda_ds = self.read_data([uri], metadata, **kwargs)
                if isinstance(kinda_ds, xr.DataArray):
                    return kinda_ds.to_dataset()
                else:
                    return kinda_ds

            def read_data(
                self, uris: List[str], metadata: SourceMetadata, **kwargs
            ) -> xr.Dataset:
                return geoda

        return MockWriteableGeoDatasetDriver

    def test_to_file(self, MockWritableDriver: Type[GeoDatasetDriver]):
        mock_driver = MockWritableDriver()

        source = GeoDatasetSource(
            name="test",
            uri="geoda.zarr",
            driver=mock_driver,
            metadata=SourceMetadata(crs=4326),
        )
        new_source = source.to_file("test")
        assert "local" in new_source.driver.filesystem.protocol
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(mock_driver) != id(new_source.driver)

    def test_to_file_override(self, MockWritableDriver: Type[GeoDatasetDriver]):
        driver1 = MockWritableDriver()
        source = GeoDatasetSource(
            name="test",
            uri="geoda.zarr",
            driver=driver1,
            metadata=SourceMetadata(crs=4326),
        )
        driver2 = MockWritableDriver(filesystem="memory")
        new_source = source.to_file("test", driver_override=driver2)
        assert new_source.driver.filesystem.protocol == "memory"
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(driver2) == id(new_source.driver)

    def test_to_file_defaults(
        self, tmp_dir: Path, geods: xr.Dataset, mock_resolver: MetaDataResolver
    ):
        class NotWritableDriver(GeoDatasetDriver):
            name: ClassVar[str] = "test_to_file_defaults"

            def read_data(self, uri: str, **kwargs) -> xr.Dataset:
                return geods

        old_path: Path = tmp_dir / "test.zarr"
        new_path: Path = tmp_dir / "temp.zarr"

        new_source: GeoDatasetSource = GeoDatasetSource(
            name="test",
            uri=str(old_path),
            driver=NotWritableDriver(metadata_resolver=mock_resolver),
        ).to_file(new_path)
        assert new_path.is_dir()  # zarr
        assert new_source.root is None
        assert new_source.driver.filesystem.protocol == ("file", "local")

    def test_raises_on_write_with_incapable_driver(
        self, MockGeoDatasetDriver: Type[GeoDatasetDriver]
    ):
        mock_driver = MockGeoDatasetDriver()
        source = GeoDatasetSource(
            name="test",
            uri="geoda.zarr",
            driver=mock_driver,
            metadata=SourceMetadata(crs=4326),
        )
        with pytest.raises(
            RuntimeError,
            match=f"driver: '{mock_driver.name}' does not support writing data",
        ):
            source.to_file("/non/existant/asdf.zarr", driver_override=mock_driver)

    @pytest.fixture()
    def writable_source(
        self, MockWritableDriver: Type[GeoDatasetDriver], geoda: xr.DataArray
    ) -> GeoDatasetSource:
        return GeoDatasetSource(
            name="test",
            uri="geoda.zarr",
            driver=MockWritableDriver(),
            metadata=SourceMetadata(crs=4326),
        )

    @pytest.mark.integration()
    def test_writes_to_netcdf(self, tmp_dir: Path, writable_source: GeoDatasetSource):
        local_driver = GeoDatasetXarrayDriver()
        local_path: Path = tmp_dir / "geods_source_writes_netcdf.nc"
        writable_source.to_file(file_path=local_path, driver_override=local_driver)
        assert local_driver.filesystem.exists(local_path)

    @pytest.mark.integration()
    def test_writes_to_netcdf_variables(
        self,
        tmp_dir: Path,
        writable_source: GeoDatasetSource,
    ):
        local_driver = GeoDatasetXarrayDriver()
        local_path = tmp_dir / "geods_source_writes_netcdf_variables.nc"
        writable_source.to_file(
            file_path=local_path,
            driver_override=local_driver,
            variables="test1",
        )
        assert local_driver.filesystem.exists(local_path)

    @pytest.mark.integration()
    def test_writes_to_zarr(self, tmp_dir: Path, writable_source: GeoDatasetSource):
        local_driver = GeoDatasetXarrayDriver()
        local_path = tmp_dir / "geods_source_writes_netcdf.zarr"
        writable_source.to_file(
            file_path=local_path,
            driver_override=local_driver,
        )
        assert local_driver.filesystem.exists(local_path)

    def test_detect_bbox(self, writable_source: GeoDatasetSource, geoda: xr.DataArray):
        geoda_expected_bbox = (-74.08, -34.58, -47.91, 10.48)
        geoda_detected_bbox = to_geographic_bbox(*writable_source.detect_bbox(geoda))
        assert np.all(np.equal(geoda_expected_bbox, geoda_detected_bbox))

    def test_detect_time_range(
        self, writable_source: GeoDatasetSource, geoda: xr.DataArray
    ):
        geoda_expected_time_range = tuple(pd.to_datetime(["01-01-2000", "12-31-2000"]))
        geoda_detected_time_range = writable_source.detect_time_range(geoda)
        assert geoda_expected_time_range == geoda_detected_time_range
