from datetime import datetime
from os.path import basename
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pytest
from pyogrio.errors import DataSourceError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem

from hydromt.data_adapter_v1.geodataframe_adapter import GeoDataFrameAdapter
from hydromt.data_catalog_v1 import DataCatalog
from hydromt.data_sources.geodataframe_data_source import GeoDataFrameDataSource
from hydromt.drivers.pyogrio_driver import PyogrioDriver
from hydromt.metadata_resolvers.convention_resolver import ConventionResolver
from hydromt.typing import ErrorHandleMethod


class TestGeoDataFrameAdapter:
    @pytest.fixture()
    def artifact_data(self):
        datacatalog = DataCatalog()
        datacatalog.from_predefined_catalogs("artifact_data")
        return datacatalog

    @pytest.fixture(scope="class")
    def example_geojson(self, geodf: gpd.GeoDataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / "test.geojson")
        geodf.to_file(uri, driver="GeoJSON")
        return uri

    @pytest.fixture()
    def example_source(self, example_geojson: str) -> GeoDataFrameDataSource:
        return GeoDataFrameDataSource(
            name="test",
            data_type="GeoDataFrame",
            uri=example_geojson,
            metadata_resolver=ConventionResolver(),
            driver=PyogrioDriver(),
        )

    @pytest.mark.integration()
    def test_get_data(
        self, geodf: gpd.GeoDataFrame, example_source: GeoDataFrameDataSource
    ):
        adapter = GeoDataFrameAdapter(source=example_source)
        gdf = adapter.get_data(list(geodf.total_bounds))
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert np.all(gdf == geodf)

        example_source.rename = {"test": "test1"}

        gdf = adapter.get_data(
            bbox=list(geodf.total_bounds),
            buffer=1000,
        )
        example_source.uri = "no_file.geojson"
        adapter = GeoDataFrameAdapter(source=example_source)
        with pytest.raises(DataSourceError):
            adapter.get_data()

    @pytest.mark.skip()  # FIXME
    def test_geodataframe_unit_attrs(self, artifact_data: DataCatalog):
        source = artifact_data.get_source(
            "gadm_level1"
        )  # TODO: fails because we have not implemented RasterDataSet
        source.attrs = {"NAME_0": {"long_name": "Country names"}}
        gdf = GeoDataFrameAdapter(source=source).get_data("gadm_level1")
        assert gdf["NAME_0"].attrs["long_name"] == "Country names"

        # gadm_level1 = {"gadm_level1": artifact_data.get_source("gadm_level1").to_dict()}
        # attrs = {"NAME_0": {"long_name": "Country names"}}
        # gadm_level1["gadm_level1"].update(dict(attrs=attrs))
        # artifact_data.from_dict(gadm_level1)
        # gadm_level1_gdf = artifact_data.get_geodataframe("gadm_level1")
        # assert gadm_level1_gdf["NAME_0"].attrs["long_name"] == "Country names"

    @pytest.mark.skip()  # FIXME
    def test_to_stac_geodataframe(self, geodf: gpd.GeoDataFrame, tmp_dir: Path):
        fn_gdf = str(tmp_dir / "test.geojson")
        geodf.to_file(fn_gdf, driver="GeoJSON")
        data_catalog = DataCatalog()  # read artifacts
        _ = data_catalog.sources  # load artifact data as fallback

        # geodataframe
        name = "gadm_level1"
        adapter = cast(
            GeoDataFrameAdapter, data_catalog.get_source(name)
        )  # TODO: Fails because we have not implemented RasterDataSet
        bbox, _ = adapter.get_bbox()
        gdf_stac_catalog = StacCatalog(id=name, description=name)
        gds_stac_item = StacItem(
            name,
            geometry=None,
            bbox=list(bbox),
            properties=adapter.meta,
            datetime=datetime(1, 1, 1),
        )
        gds_stac_asset = StacAsset(str(adapter.path))
        gds_base_name = basename(adapter.path)
        gds_stac_item.add_asset(gds_base_name, gds_stac_asset)

        gdf_stac_catalog.add_item(gds_stac_item)
        outcome = cast(
            StacCatalog, adapter.to_stac_catalog(on_error=ErrorHandleMethod.RAISE)
        )
        assert gdf_stac_catalog.to_dict() == outcome.to_dict()  # type: ignore
        adapter.crs = -3.14  # manually create an invalid adapter by deleting the crs
        assert adapter.to_stac_catalog(on_error=ErrorHandleMethod.SKIP) is None
