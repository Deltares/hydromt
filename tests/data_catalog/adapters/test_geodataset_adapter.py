import numpy as np
import pytest
import xarray as xr

from hydromt._typing import SourceMetadata
from hydromt.data_catalog.adapters.geodataset import GeoDatasetAdapter
from hydromt.gis._gis_utils import _parse_geom_bbox_buffer


class TestGeoDatasetAdapter:
    @pytest.fixture()
    def example_geo_ds(self, geoda: xr.DataArray) -> xr.Dataset:
        geoda.vector.set_crs(4326)
        return geoda.to_dataset()

    def test_transform_data_bbox(self, example_geo_ds: xr.Dataset):
        adapter = GeoDatasetAdapter()
        mask = _parse_geom_bbox_buffer(bbox=example_geo_ds.vector.bounds)
        ds = adapter.transform(
            example_geo_ds,
            metadata=SourceMetadata(),
            mask=mask,
        )
        assert np.all(ds == example_geo_ds)

    def test_transform_data_mask(self, example_geo_ds: xr.Dataset):
        adapter = GeoDatasetAdapter()
        ds = adapter.transform(example_geo_ds, metadata=SourceMetadata())
        assert np.all(ds == example_geo_ds)

    # TODO: https://github.com/Deltares/hydromt/issues/802
    # def test_transform_nodata(self, example_geo_ds: xr.Dataset):
    #     adapter = GeoDatasetAdapter()
    #     mask = parse_geom_bbox_buffer(bbox=(40, 50, 41, 51))
    # with pytest.raises(NoDataException):
    #     adapter.transform(example_geo_ds, metadata=SourceMetadata(), mask=mask)
    # ds = adapter.transform(
    #     example_geo_ds,
    #     metadata=SourceMetadata(),
    #     mask=mask,
    #     handle_nodata=NoDataStrategy.IGNORE,
    # )
    # assert ds is None
