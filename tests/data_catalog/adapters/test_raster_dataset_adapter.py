import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import box

from hydromt.data_catalog.adapters.rasterdataset import RasterDatasetAdapter
from hydromt.error import NoDataException, NoDataStrategy
from hydromt.typing import SourceMetadata


class TestRasterDatasetAdapter:
    @pytest.fixture
    def example_raster_ds(self):
        nx, ny, nt = 50, 50, 3
        xmin, xmax = -100, -99
        ymin, ymax = 42, 43
        temp = 15 + 8 * np.random.randn(nx, ny, nt)
        precip = 10 * np.random.rand(nx, ny, nt)

        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymax, ymin, ny)  # decreasing, top-left origin

        ds = xr.Dataset(
            {
                "temperature": (["y", "x", "time"], temp),
                "precipitation": (["y", "x", "time"], precip),
            },
            coords={
                "x": x,
                "y": y,
                "time": pd.date_range("2014-09-06", periods=nt),
                "reference_time": pd.Timestamp("2014-09-05"),
            },
        )
        ds.raster.set_crs(4326)
        return ds

    def test_transform_data_bbox(self, example_raster_ds: xr.Dataset):
        adapter = RasterDatasetAdapter()
        ds = adapter.transform(
            example_raster_ds,
            metadata=SourceMetadata(),
            mask=gpd.GeoSeries(box(*example_raster_ds.raster.bounds)).set_crs(4326),
        )
        assert np.all(ds == example_raster_ds)

    def test_transform_no_mask_no_buffer(self, example_raster_ds: xr.Dataset):
        adapter = RasterDatasetAdapter()

        ds = adapter.transform(
            example_raster_ds,
            metadata=SourceMetadata(),
            mask=None,
            buffer=0,
        )

        # Should be identical to the input
        assert ds is not None
        assert ds.raster.bounds == example_raster_ds.raster.bounds
        assert ds.sizes == example_raster_ds.sizes

    def test_transform_with_mask_no_buffer(self, example_raster_ds: xr.Dataset):
        adapter = RasterDatasetAdapter()

        # construct a small mask inside the dataset bounds
        xmin, ymin, xmax, ymax = example_raster_ds.raster.bounds
        xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2
        sizex, sizey = (xmax - xmin) / 4, (ymax - ymin) / 4
        mask_geom = box(xmid - sizex, ymid - sizey, xmid + sizex, ymid + sizey)
        mask = gpd.GeoSeries([mask_geom], crs=4326)

        ds = adapter.transform(
            example_raster_ds,
            metadata=SourceMetadata(),
            mask=mask,
            buffer=0,
        )

        # Should be cropped to mask → bounds smaller than the original dataset & almost eq to mask bounds, but aligned with the raster
        assert ds is not None
        orig_bounds = example_raster_ds.raster.bounds
        returned_bounds = ds.raster.bounds
        mask_bounds = mask.total_bounds
        dx, dy = example_raster_ds.raster.res
        dx, dy = abs(dx), abs(dy)

        assert returned_bounds[0] > orig_bounds[0]
        assert np.isclose(returned_bounds[0], mask_bounds[0], atol=dx)

        assert returned_bounds[1] > orig_bounds[1]
        assert np.isclose(returned_bounds[1], mask_bounds[1], atol=dy)

        assert returned_bounds[2] < orig_bounds[2]
        assert np.isclose(returned_bounds[2], mask_bounds[2], atol=dx)

        assert returned_bounds[3] < orig_bounds[3]
        assert np.isclose(returned_bounds[3], mask_bounds[3], atol=dy)

    def test_transform_with_mask_and_buffer(self, example_raster_ds: xr.Dataset):
        adapter = RasterDatasetAdapter()

        xmin, ymin, xmax, ymax = example_raster_ds.raster.bounds
        xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2
        sizex, sizey = (xmax - xmin) / 4, (ymax - ymin) / 4
        mask_geom = box(xmid - sizex, ymid - sizey, xmid + sizex, ymid + sizey)
        mask = gpd.GeoSeries([mask_geom], crs=4326)

        buffer_cells = 2
        ds = adapter.transform(
            example_raster_ds,
            metadata=SourceMetadata(),
            mask=mask,
            buffer=buffer_cells,
        )

        assert ds is not None
        orig_bounds = example_raster_ds.raster.bounds
        returned_bounds = ds.raster.bounds
        mask_bounds = mask.total_bounds
        dx, dy = example_raster_ds.raster.res
        dx, dy = abs(dx), abs(dy)

        # Mask smaller than original
        assert mask_bounds[0] > orig_bounds[0]
        assert mask_bounds[1] > orig_bounds[1]
        assert mask_bounds[2] < orig_bounds[2]
        assert mask_bounds[3] < orig_bounds[3]

        # Buffer in world coords = mask_bound +- buffer_cells * resolution
        expected_minx = mask_bounds[0] - buffer_cells * dx
        expected_miny = mask_bounds[1] - buffer_cells * dy
        expected_maxx = mask_bounds[2] + buffer_cells * dx
        expected_maxy = mask_bounds[3] + buffer_cells * dy

        # Allow 1 cell tolerance due to alignment rounding
        assert np.isclose(returned_bounds[0], expected_minx, atol=dx)
        assert np.isclose(returned_bounds[1], expected_miny, atol=dy)
        assert np.isclose(returned_bounds[2], expected_maxx, atol=dx)
        assert np.isclose(returned_bounds[3], expected_maxy, atol=dy)

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
