import numpy as np
import pytest
import xarray as xr

from hydromt._typing import NoDataException, NoDataStrategy
from hydromt.data_adapter.rasterdataset import RasterDatasetAdapter
from hydromt.data_source import SourceMetadata


class TestRasterDatasetAdapter:
    @pytest.fixture()
    def example_raster_ds(self, rasterds: xr.Dataset):
        rasterds.raster.set_crs(4326)
        return rasterds

    def test_get_data_bbox(self, example_raster_ds: xr.Dataset):
        adapter = RasterDatasetAdapter()
        ds = adapter.transform(
            example_raster_ds,
            metadata=SourceMetadata(),
            bbox=example_raster_ds.raster.bounds,
        )
        assert np.all(ds == example_raster_ds)

    def test_get_data_mask(self, example_raster_ds: xr.Dataset):
        adapter = RasterDatasetAdapter()
        geom = example_raster_ds.raster.box.set_crs(4326)
        ds = adapter.transform(example_raster_ds, metadata=SourceMetadata(), mask=geom)
        assert np.all(ds == example_raster_ds)

    def test_nodata(self, example_raster_ds: xr.Dataset):
        adapter = RasterDatasetAdapter()
        with pytest.raises(NoDataException):
            adapter.transform(
                example_raster_ds, metadata=SourceMetadata(), bbox=(40, 50, 41, 51)
            )
        ds = adapter.transform(
            example_raster_ds,
            metadata=SourceMetadata(),
            bbox=(40, 50, 41, 51),
            handle_nodata=NoDataStrategy.IGNORE,
        )
        assert ds is None
