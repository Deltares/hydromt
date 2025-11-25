from pathlib import Path
from typing import Type

import numpy as np
import pytest
import xarray as xr

from hydromt.data_catalog.adapters import GeoDatasetAdapter
from hydromt.data_catalog.drivers import GeoDatasetDriver, GeoDatasetXarrayDriver
from hydromt.data_catalog.sources import GeoDatasetSource
from hydromt.data_catalog.uri_resolvers import URIResolver
from hydromt.gis.gis_utils import _to_geographic_bbox
from hydromt.typing import SourceMetadata
from hydromt.typing.type_def import TimeRange


class TestGeoDatasetSource:
    def test_read_data(
        self,
        geoda: xr.DataArray,
        MockGeoDatasetDriver: type[GeoDatasetDriver],
        mock_geo_ds_adapter: GeoDatasetAdapter,
        mock_resolver: URIResolver,
        managed_tmp_path: Path,
    ):
        geoda = geoda.to_dataset()
        source = GeoDatasetSource(
            root=".",
            name="geoda.zarr",
            driver=MockGeoDatasetDriver(),
            data_adapter=mock_geo_ds_adapter,
            uri_resolver=mock_resolver,
            uri=str(managed_tmp_path / "geoda.zarr"),
        )
        read_data = source.read_data()
        assert read_data.equals(geoda)

    @pytest.fixture
    def writable_source(
        self,
        MockGeoDatasetReadOnlyDriver: Type[GeoDatasetDriver],
        mock_resolver: URIResolver,
    ) -> GeoDatasetSource:
        return GeoDatasetSource(
            name="test",
            uri="geoda.zarr",
            driver=MockGeoDatasetReadOnlyDriver(),
            uri_resolver=mock_resolver,
            metadata=SourceMetadata(crs=4326),
        )

    @pytest.mark.integration
    def test_writes_to_netcdf(
        self, managed_tmp_path: Path, writable_source: GeoDatasetSource
    ):
        local_driver = GeoDatasetXarrayDriver()
        local_path = managed_tmp_path / "geods_source_writes_netcdf.nc"
        writable_source.to_file(file_path=local_path, driver_override=local_driver)
        assert local_driver.filesystem.get_fs().exists(local_path)

    @pytest.mark.integration
    def test_writes_to_netcdf_variables(
        self,
        managed_tmp_path: Path,
        writable_source: GeoDatasetSource,
    ):
        local_driver = GeoDatasetXarrayDriver()
        local_path = managed_tmp_path / "geods_source_writes_netcdf_variables.nc"
        writable_source.to_file(
            file_path=local_path,
            driver_override=local_driver,
            variables="test1",
        )
        assert local_driver.filesystem.get_fs().exists(local_path)

    @pytest.mark.integration
    def test_writes_to_zarr(
        self, managed_tmp_path: Path, writable_source: GeoDatasetSource
    ):
        local_driver = GeoDatasetXarrayDriver()
        local_path = managed_tmp_path / "geods_source_writes_netcdf.zarr"
        writable_source.to_file(
            file_path=local_path,
            driver_override=local_driver,
        )
        assert local_driver.filesystem.get_fs().exists(local_path)

    def test_detect_bbox(self, writable_source: GeoDatasetSource, geoda: xr.DataArray):
        geoda_expected_bbox = (-74.08, -34.58, -47.91, 10.48)
        geoda_detected_bbox = _to_geographic_bbox(
            *writable_source._detect_bbox(ds=geoda)
        )
        assert np.all(np.equal(geoda_expected_bbox, geoda_detected_bbox))

    def test_detect_time_range(
        self, writable_source: GeoDatasetSource, geoda: xr.DataArray
    ):
        geoda_expected_time_range = TimeRange(start="01-01-2000", end="12-31-2000")
        geoda_detected_time_range = writable_source._detect_time_range(ds=geoda)
        assert geoda_expected_time_range == geoda_detected_time_range

    @pytest.mark.parametrize(
        ("uri", "expected_driver"),
        [
            ("test_data.csv", "geodataset_vector"),
            ("test_data.zarr", "geodataset_xarray"),
            ("test_data.fake_suffix", "geodataset_vector"),
        ],
    )
    def test_infer_default_driver(self, uri, expected_driver):
        assert GeoDatasetSource._infer_default_driver(uri) == expected_driver
