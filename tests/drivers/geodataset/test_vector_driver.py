"""Tests the RasterXarray driver."""

from pathlib import Path
from typing import Optional

import pytest
from geopandas import GeoDataFrame
from pytest_mock import MockerFixture

from hydromt.drivers import GeoDatasetVectorDriver
from hydromt.drivers.preprocessing import round_latlon
from hydromt.gis import vector
from hydromt.io.readers import open_geodataset
from hydromt.metadata_resolver.convention_resolver import ConventionResolver
from hydromt.metadata_resolver.metadata_resolver import MetaDataResolver


class TestGeoDatasetVectorDriver:
    def test_calls_preprocess(self, mocker: MockerFixture):
        mock_geods_open: mocker.MagicMock = mocker.patch(
            "hydromt.drivers.geodataset.vector_driver.open_geodataset",
            spec=open_geodataset,
        )
        mock_geods_open.return_value = GeoDataFrame()

        class FakeMetadataResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return [uri]

        uri: str = "file.geojson"
        driver = GeoDatasetVectorDriver(
            metadata_resolver=FakeMetadataResolver(),
            options={"preprocess": "round_latlon"},
        )
        res: Optional[GeoDataFrame] = driver.read(
            uri,
            variables=["var1"],
        )
        assert res is not None
        call_args = mock_geods_open.call_args

        assert call_args[1]["fn_locs"] == uri  # first arg
        assert call_args[1].get("preprocess") == round_latlon
        assert res.sizes == {}  # empty dataframe

        assert (
            driver.options.get("preprocess") == "round_latlon"
        )  # test does not consume property

    def test_write_raises(self):
        driver = GeoDatasetVectorDriver()
        with pytest.raises(NotImplementedError):
            driver.write()  # type: ignore

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
        assert ds.equals(res)

    def test_calls_open_geodataset(self, mocker: MockerFixture):
        mock_geods_open: mocker.MagicMock = mocker.patch(
            "hydromt.drivers.geodataset.vector_driver.open_geodataset",
            spec=open_geodataset,
        )
        mock_geods_open.return_value = GeoDataFrame()

        class FakeMetadataResolver(MetaDataResolver):
            def resolve(self, uri: str, *args, **kwargs):
                return [uri]

        uri: str = "file.geojson"
        driver = GeoDatasetVectorDriver(metadata_resolver=FakeMetadataResolver())
        _ = driver.read(
            uri,
        )
        assert mock_geods_open.call_count == 1
