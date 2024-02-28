import numpy as np
import pytest
import xarray as xr

from hydromt._typing import NoDataException, NoDataStrategy
from hydromt.data_adapter.rasterdataset import RasterDatasetAdapter
from hydromt.data_sources.rasterdataset import RasterDataSource
from hydromt.drivers.rasterdataset_driver import RasterDatasetDriver
from hydromt.metadata_resolvers.metadata_resolver import MetaDataResolver


class TestRasterDatasetAdapter:
    @pytest.fixture()
    def example_source(
        self,
        mock_raster_ds_driver: RasterDatasetDriver,
        mock_resolver: MetaDataResolver,
    ):
        return RasterDataSource(
            name="example_source",
            driver=mock_raster_ds_driver,
            metadata_resolver=mock_resolver,
            uri="my_uri.zarr",
            crs=4326,
        )

    @pytest.mark.integration()
    def test_get_data(self, example_source: RasterDataSource, rasterds: xr.Dataset):
        adapter = RasterDatasetAdapter(source=example_source)
        ds = adapter.get_data(bbox=rasterds.raster.bounds)
        assert np.all(ds == rasterds)
        geom = rasterds.raster.box.set_crs(4326)
        ds = adapter.get_data(geom=geom)
        assert np.all(ds == rasterds)
        with pytest.raises(NoDataException):
            adapter.get_data(bbox=(40, 50, 41, 51))
        ds = adapter.get_data(
            bbox=(40, 50, 41, 51), handle_nodata=NoDataStrategy.IGNORE
        )
        assert ds is None
