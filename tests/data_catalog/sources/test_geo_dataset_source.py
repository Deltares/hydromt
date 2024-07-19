from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydromt._typing import SourceMetadata
from hydromt.data_catalog.adapters import GeoDatasetAdapter
from hydromt.data_catalog.drivers import GeoDatasetDriver, GeoDatasetXarrayDriver
from hydromt.data_catalog.sources import GeoDatasetSource
from hydromt.data_catalog.uri_resolvers import URIResolver
from hydromt.gis.gis_utils import to_geographic_bbox


class TestGeoDatasetSource:
    def test_read_data(
        self,
        geoda: xr.DataArray,
        mock_geo_ds_driver: GeoDatasetDriver,
        mock_geo_ds_adapter: GeoDatasetAdapter,
        mock_resolver: URIResolver,
        tmp_dir: Path,
    ):
        geoda = geoda.to_dataset()
        source = GeoDatasetSource(
            root=".",
            name="geoda.zarr",
            driver=mock_geo_ds_driver,
            data_adapter=mock_geo_ds_adapter,
            uri_resolver=mock_resolver,
            uri=str(tmp_dir / "geoda.zarr"),
        )
        read_data = source.read_data()
        assert read_data.equals(geoda)

    @pytest.fixture()
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
