"""Tests the RasterXarray driver."""

from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture
from xarray import Dataset

from hydromt.data_catalog.drivers import GeoDatasetVectorDriver, preprocessing
from hydromt.data_catalog.drivers.xarray_options import XarrayDriverOptions
from hydromt.gis import vector
from hydromt.readers import open_geodataset


class TestGeoDatasetVectorDriver:
    def test_calls_preprocess(self, mocker: MockerFixture):
        mock_geods_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.vector_driver.open_geodataset",
            spec=open_geodataset,
        )
        mock_ds = Dataset(
            coords={"time": pd.date_range("2023-01-01", periods=3)},
            data_vars={"var1": ("time", [1, 2, 2])},
        )
        mock_geods_open.return_value = mock_ds

        # Patch PREPROCESSORS so every preprocessor just returns its input
        mock_preprocessor = mocker.MagicMock(side_effect=lambda data: data)

        mocker.patch.object(
            preprocessing,
            "PREPROCESSORS",
            {name: mock_preprocessor for name in preprocessing.PREPROCESSORS.keys()},
        )

        uris: List[str] = ["file.geojson"]
        driver = GeoDatasetVectorDriver(
            options=XarrayDriverOptions(preprocess="remove_duplicates"),
        )
        res: Optional[Dataset] = driver.read(uris)
        assert res is not None
        mock_preprocessor.assert_called_once_with(mock_ds)

    def test_write_raises(self):
        driver = GeoDatasetVectorDriver()
        with pytest.raises(NotImplementedError):
            driver.write("fake_path.zarr", MagicMock())  # type: ignore

    @pytest.fixture
    def example_vector_geods(self, geodf, managed_tmp_path: Path) -> Path:
        gdf_path = managed_tmp_path / "test.geojson"
        geodf.to_file(gdf_path, driver="GeoJSON")
        return gdf_path

    def test_read(self, geodf, example_vector_geods: Path):
        res = GeoDatasetVectorDriver().read([str(example_vector_geods)])
        ds = vector.GeoDataset.from_gdf(geodf)
        assert res is not None
        assert ds.equals(res)

    def test_raises_on_multiple_uris(self):
        with pytest.raises(
            ValueError,
            match="GeodatasetVectorDriver only supports reading from one URI per source",
        ):
            _ = GeoDatasetVectorDriver().read(["one.zarr", "two.txt"])

    def test_calls_open_geodataset(self, mocker: MockerFixture):
        mock_geods_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.vector_driver.open_geodataset",
            spec=open_geodataset,
        )
        mock_geods_open.return_value = Dataset()

        uris: List[str] = ["file.geojson"]
        driver = GeoDatasetVectorDriver()
        _ = driver.read(uris)
        assert mock_geods_open.call_count == 1
