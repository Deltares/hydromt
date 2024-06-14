from pathlib import Path
from typing import Type

import numpy as np
import pytest
import xarray as xr

from hydromt._typing import SourceMetadata
from hydromt.data_catalog.adapters import RasterDatasetAdapter
from hydromt.data_catalog.drivers import RasterDatasetDriver
from hydromt.data_catalog.sources import RasterDatasetSource
from hydromt.gis.gis_utils import to_geographic_bbox


class TestRasterDatasetSource:
    def test_read_data(
        self,
        raster_ds: xr.Dataset,
        mock_raster_ds_driver: RasterDatasetDriver,
        mock_raster_ds_adapter: RasterDatasetAdapter,
        tmp_dir: Path,
    ):
        source = RasterDatasetSource(
            root=".",
            name="example_rasterds",
            driver=mock_raster_ds_driver,
            data_adapter=mock_raster_ds_adapter,
            uri=str(tmp_dir / "rasterds.zarr"),
        )
        assert raster_ds == source.read_data()

    @pytest.fixture()
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
        rioda_detected_bbox = to_geographic_bbox(*writable_source.detect_bbox(rioda))

        assert np.all(np.equal(rioda_expected_bbox, rioda_detected_bbox))
