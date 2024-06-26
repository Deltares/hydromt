"""Tests the RasterXarray driver."""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from xarray import Dataset

from hydromt.data_catalog.drivers import GeoDatasetVectorDriver
from hydromt.data_catalog.uri_resolvers.convention_resolver import ConventionResolver
from hydromt.data_catalog.uri_resolvers.metadata_resolver import MetaDataResolver
from hydromt.gis import vector
from hydromt.io.readers import open_geodataset


class TestGeoDatasetVectorDriver:
    def test_calls_preprocess(self, mocker: MockerFixture):
        mock_geods_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.vector_driver.open_geodataset",
            spec=open_geodataset,
        )
        mock_geods_open.return_value = Dataset()

        mock_preprocess: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.vector_driver.PREPROCESSORS",
            spec=dict,
        )

        mocked_function = MagicMock(return_value=Dataset())
        mock_preprocess.get.return_value = mocked_function

        class FakeMetadataResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return [uri]

        uri: str = "file.geojson"
        driver = GeoDatasetVectorDriver(
            metadata_resolver=FakeMetadataResolver(),
            options={"preprocess": "remove_duplicates"},
        )
        res: Optional[Dataset] = driver.read(
            uri,
            variables=["var1"],
        )
        assert res is not None
        call_args = mock_geods_open.call_args

        assert call_args[1]["location_data_path"] == uri  # first arg
        assert mocked_function.call_count == 1

    def test_write_raises(self):
        driver = GeoDatasetVectorDriver()
        with pytest.raises(NotImplementedError):
            driver.write("fake_path.zarr", MagicMock())  # type: ignore

    @pytest.fixture()
    def example_vector_geods(self, geodf, tmp_dir: Path) -> Path:
        base = Path(tmp_dir)
        gdf_path = base / "test.geojson"
        geodf.to_file(gdf_path, driver="GeoJSON")
        return gdf_path

    def test_read(self, geodf, example_vector_geods: Path):
        res = GeoDatasetVectorDriver(metadata_resolver=ConventionResolver()).read(
            str(example_vector_geods)
        )
        ds = vector.GeoDataset.from_gdf(geodf)
        assert res is not None
        assert ds.equals(res)

    def test_raises_on_multiple_uris(self):
        with pytest.raises(
            ValueError,
            match="GeodatasetVectorDriver only supports reading from one URI per source",
        ):
            _ = GeoDatasetVectorDriver().read_data(["one.zarr", "two.txt"])

    def test_calls_open_geodataset(self, mocker: MockerFixture):
        mock_geods_open: mocker.MagicMock = mocker.patch(
            "hydromt.data_catalog.drivers.geodataset.vector_driver.open_geodataset",
            spec=open_geodataset,
        )
        mock_geods_open.return_value = Dataset()

        class FakeMetadataResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return [uri]

        uri: str = "file.geojson"
        driver = GeoDatasetVectorDriver(metadata_resolver=FakeMetadataResolver())
        _ = driver.read(uri)
        assert mock_geods_open.call_count == 1
