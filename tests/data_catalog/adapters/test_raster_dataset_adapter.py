import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely import box

from hydromt._typing import NoDataException, NoDataStrategy, SourceMetadata
from hydromt.data_catalog.adapters.rasterdataset import RasterDatasetAdapter


class TestRasterDatasetAdapter:
    @pytest.fixture()
    def example_raster_ds(self, raster_ds: xr.Dataset):
        raster_ds.raster.set_crs(4326)
        return raster_ds

    def test_transform_data_bbox(self, example_raster_ds: xr.Dataset):
        adapter = RasterDatasetAdapter()
        ds = adapter.transform(
            example_raster_ds,
            metadata=SourceMetadata(),
            mask=gpd.GeoSeries(box(*example_raster_ds.raster.bounds)).set_crs(4326),
        )
        assert np.all(ds == example_raster_ds)

    def test_transform_data_mask(self, example_raster_ds: xr.Dataset):
        adapter = RasterDatasetAdapter()
        geom = example_raster_ds.raster.box.set_crs(4326)
        ds = adapter.transform(example_raster_ds, SourceMetadata(), mask=geom)
        assert np.all(ds == example_raster_ds)

    def test_transform_nodata(self, example_raster_ds: xr.Dataset):
        adapter = RasterDatasetAdapter()
        mask = gpd.GeoSeries.from_wkt(
            ["POLYGON ((40 50, 41 50, 41 51, 40 51, 40 50))"]
        ).set_crs(4326)
        with pytest.raises(NoDataException):
            adapter.transform(example_raster_ds, metadata=SourceMetadata(), mask=mask)
        ds = adapter.transform(
            example_raster_ds,
            metadata=SourceMetadata(),
            mask=mask,
            handle_nodata=NoDataStrategy.IGNORE,
        )
        assert ds is None
