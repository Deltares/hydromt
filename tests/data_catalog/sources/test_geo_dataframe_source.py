from datetime import datetime
from os.path import basename
from pathlib import Path
from typing import cast
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pytest
from pydantic import ValidationError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem

from hydromt.data_catalog import DataCatalog
from hydromt.data_catalog.adapters.geodataframe import GeoDataFrameAdapter
from hydromt.data_catalog.drivers import GeoDataFrameDriver, PyogrioDriver
from hydromt.data_catalog.sources.geodataframe import GeoDataFrameSource
from hydromt.data_catalog.uri_resolvers import URIResolver
from hydromt.error import NoDataException, NoDataStrategy


class TestGeoDataFrameSource:
    @pytest.fixture
    def artifact_data(self):
        datacatalog = DataCatalog()
        datacatalog.from_predefined_catalogs("artifact_data")
        return datacatalog

    @pytest.fixture(scope="class")
    def example_geojson(self, geodf: gpd.GeoDataFrame, managed_tmp_path: Path) -> str:
        uri = managed_tmp_path / f"{uuid4().hex}.geojson"
        geodf.to_file(uri, driver="GeoJSON")
        return str(uri)

    def test_raises_on_invalid_fields(
        self,
        mock_gdf_adapter: GeoDataFrameAdapter,
        MockGeoDataFrameDriver: type[GeoDataFrameDriver],
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GeoDataFrameSource(
                root=".",
                name="name",
                uri="uri",
                data_adapter=mock_gdf_adapter,
                driver=MockGeoDataFrameDriver(),
                foo="bar",
            )

    def test_get_data_query_params(
        self,
        geodf: gpd.GeoDataFrame,
        mock_resolver: URIResolver,
        MockGeoDataFrameDriver: type[GeoDataFrameDriver],
        mock_gdf_adapter: GeoDataFrameAdapter,
    ):
        data_source = GeoDataFrameSource(
            root=".",
            name="geojsonfile",
            data_type="GeoDataFrame",
            driver=MockGeoDataFrameDriver(),
            data_adapter=mock_gdf_adapter,
            uri_resolver=mock_resolver,
            uri="testuri",
        )
        gdf1 = data_source.read_data(bbox=list(geodf.total_bounds), buffer=1000)
        assert isinstance(gdf1, gpd.GeoDataFrame)
        assert np.all(gdf1 == geodf)

    @pytest.mark.integration
    def test_get_data(self, geodf: gpd.GeoDataFrame, example_geojson: str):
        source = GeoDataFrameSource(
            name="test",
            uri=example_geojson,
            data_adapter=GeoDataFrameAdapter(),
            driver=PyogrioDriver(),
        )
        gdf = source.read_data(bbox=list(geodf.total_bounds))
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert np.all(gdf == geodf)

    @pytest.mark.integration
    def test_get_data_rename(self, geodf: gpd.GeoDataFrame, example_geojson: str):
        source = GeoDataFrameSource(
            name="test",
            uri=example_geojson,
            data_adapter=GeoDataFrameAdapter(rename={"city": "ciudad"}),
            driver=PyogrioDriver(),
        )
        gdf = source.read_data(
            bbox=list(geodf.total_bounds),
            buffer=1000,
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "ciudad" in gdf.columns
        assert "city" not in gdf.columns

    @pytest.mark.integration
    def test_get_data_not_found(self):
        source = GeoDataFrameSource(
            name="test",
            uri="no_file.geojson",
            data_adapter=GeoDataFrameAdapter(),
            driver=PyogrioDriver(),
        )
        with pytest.raises(NoDataException):
            source.read_data()

    def test_instantiate_mixed_objects(self):
        GeoDataFrameSource(
            name="test",
            uri="points.geojson",
            driver=PyogrioDriver(),
            data_adapter={"unit_add": {"geoattr": 1.0}},
        )

    def test_geodataframe_unit_attrs(self, artifact_data: DataCatalog):
        source = artifact_data.get_source("gadm_level1")
        source.metadata.attrs = {"NAME_0": {"long_name": "Country names"}}
        gdf = source.read_data()
        assert gdf.attrs["NAME_0"]["long_name"] == "Country names"

    def test_to_stac_geodataframe(
        self, geodf: gpd.GeoDataFrame, managed_tmp_path: Path
    ):
        gdf_path = managed_tmp_path / "test.geojson"
        geodf.to_file(gdf_path, driver="GeoJSON")
        data_catalog = DataCatalog()  # read artifacts
        _ = data_catalog.sources  # load artifact data as fallback

        # geodataframe
        name = "gadm_level1"
        adapter = cast(GeoDataFrameAdapter, data_catalog.get_source(name))
        bbox, _ = adapter.get_bbox()
        gdf_stac_catalog = StacCatalog(id=name, description=name)
        gds_stac_item = StacItem(
            name,
            geometry=None,
            bbox=list(bbox),
            properties=adapter.metadata,
            datetime=datetime(1, 1, 1),
        )
        gds_stac_asset = StacAsset(str(adapter.uri))
        gds_base_name = basename(adapter.uri)
        gds_stac_item.add_asset(gds_base_name, gds_stac_asset)

        gdf_stac_catalog.add_item(gds_stac_item)
        outcome = cast(
            StacCatalog, adapter.to_stac_catalog(handle_nodata=NoDataStrategy.RAISE)
        )
        assert gdf_stac_catalog.to_dict() == outcome.to_dict()  # type: ignore
        adapter.metadata.crs = (
            -3.14
        )  # manually create an invalid adapter by deleting the crs
        assert adapter.to_stac_catalog(handle_nodata=NoDataStrategy.IGNORE) is None

        gdf_path = managed_tmp_path / "test.geojson"
        geodf.to_file(gdf_path, driver="GeoJSON")
        source = GeoDataFrameSource(name="test_data", uri=str(gdf_path))

        with pytest.raises(
            RuntimeError,
            match="Unknown extension: .geojson, cannot determine media type",
        ):
            source.to_stac_catalog()

    @pytest.mark.parametrize(
        ("uri", "expected_driver"),
        [
            ("test_data.csv", "geodataframe_table"),
            ("test_data.fgb", "pyogrio"),
            ("test_data.fake_suffix", "pyogrio"),
        ],
    )
    def test_infer_default_driver(self, uri, expected_driver):
        assert GeoDataFrameSource._infer_default_driver(uri) == expected_driver
