from pathlib import Path
from typing import Type

import numpy as np
import pytest
import xarray as xr

from hydromt._typing import SourceMetadata
from hydromt.data_catalog.adapters import RasterDatasetAdapter
from hydromt.data_catalog.drivers import RasterDatasetDriver
from hydromt.data_catalog.sources import RasterDatasetSource
from hydromt.data_catalog.uri_resolvers import URIResolver
from hydromt.gis.gis_utils import _to_geographic_bbox


class TestRasterDatasetSource:
    def test_read_data(
        self,
        raster_ds: xr.Dataset,
        MockRasterDatasetDriver: type[RasterDatasetDriver],
        mock_raster_ds_adapter: RasterDatasetAdapter,
        mock_resolver: URIResolver,
        managed_tmp_path: Path,
    ):
        source = RasterDatasetSource(
            root=".",
            name="example_rasterds",
            driver=MockRasterDatasetDriver(),
            data_adapter=mock_raster_ds_adapter,
            uri_resolver=mock_resolver,
            uri=str(managed_tmp_path / "rasterds.zarr"),
        )
        assert raster_ds == source.read_data()

    @pytest.fixture
    def writable_source(
        self, MockRasterDatasetReadOnlyDriver: Type[RasterDatasetDriver]
    ) -> RasterDatasetSource:
        return RasterDatasetSource(
            name="test",
            uri="raster.zarr",
            driver=MockRasterDatasetReadOnlyDriver(),
            metadata=SourceMetadata(crs=4326),
        )

    def test_detect_extent(
        self, writable_source: RasterDatasetSource, rioda: xr.DataArray
    ):
        rioda_expected_bbox = (3.0, -11.0, 6.0, -9.0)
        rioda_detected_bbox = _to_geographic_bbox(
            *writable_source._detect_bbox(ds=rioda)
        )

        assert np.all(np.equal(rioda_expected_bbox, rioda_detected_bbox))

    @pytest.mark.parametrize(
        ("uri", "expected_driver"),
        [
            ("test_data.tif", "rasterio"),
            ("test_data.nc", "raster_xarray"),
            ("test_data.zarr", "raster_xarray"),
            ("test_data.fake_suffix", "rasterio"),
        ],
    )
    def test_infer_default_driver(self, uri, expected_driver):
        assert RasterDatasetSource._infer_default_driver(uri) == expected_driver
