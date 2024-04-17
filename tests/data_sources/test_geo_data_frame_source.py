from datetime import datetime
from os.path import basename
from pathlib import Path
from typing import Type, cast
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pytest
from pydantic import ValidationError
from pystac import Asset as StacAsset
from pystac import Catalog as StacCatalog
from pystac import Item as StacItem

from hydromt._typing import StrPath
from hydromt._typing.error import ErrorHandleMethod
from hydromt.data_adapter.geodataframe import GeoDataFrameAdapter
from hydromt.data_catalog import DataCatalog
from hydromt.data_source.geodataframe import GeoDataFrameSource
from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver
from hydromt.drivers.pyogrio_driver import PyogrioDriver
from hydromt.metadata_resolver.convention_resolver import ConventionResolver


class TestGeoDataFrameSource:
    @pytest.fixture()
    def artifact_data(self):
        datacatalog = DataCatalog()
        datacatalog.from_predefined_catalogs("artifact_data")
        return datacatalog

    @pytest.fixture(scope="class")
    def example_geojson(self, geodf: gpd.GeoDataFrame, tmp_dir: Path) -> str:
        uri = str(tmp_dir / f"{uuid4().hex}.geojson")
        geodf.to_file(uri, driver="GeoJSON")
        return uri

    def test_validators(self, mock_geodataframe_adapter: GeoDataFrameAdapter):
        with pytest.raises(ValidationError) as e_info:
            GeoDataFrameSource(
                root=".",
                name="name",
                data_type="GeoDataFrame",
                uri="uri",
                data_adapter=mock_geodataframe_adapter,
                driver="does not exist",
            )

        assert e_info.value.error_count() == 1
        error_driver = next(
            filter(lambda e: e["loc"] == ("driver",), e_info.value.errors())
        )
        assert error_driver["type"] == "model_type"

    def test_raises_on_invalid_fields(
        self,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
        mock_geodf_driver: GeoDataFrameDriver,
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GeoDataFrameSource(
                root=".",
                name="name",
                uri="uri",
                data_adapter=mock_geodataframe_adapter,
                driver=mock_geodf_driver,
                foo="bar",
            )

    def test_model_validate(
        self,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
    ):
        GeoDataFrameSource.model_validate(
            {
                "name": "geojsonfile",
                "data_type": "GeoDataFrame",
                "driver": mock_geodf_driver,
                "data_adapter": mock_geodataframe_adapter,
                "uri": "test_uri",
            }
        )
        with pytest.raises(
            ValidationError, match="'data_type' must be 'GeoDataFrame'."
        ):
            GeoDataFrameSource.model_validate(
                {
                    "name": "geojsonfile",
                    "data_type": "DifferentDataType",
                    "driver": mock_geodf_driver,
                    "data_adapter": mock_geodataframe_adapter,
                    "uri": "test_uri",
                }
            )

    def test_get_data_query_params(
        self,
        geodf: gpd.GeoDataFrame,
        mock_geodf_driver: GeoDataFrameDriver,
        mock_geodataframe_adapter: GeoDataFrameAdapter,
    ):
        data_source = GeoDataFrameSource(
            root=".",
            name="geojsonfile",
            data_type="GeoDataFrame",
            driver=mock_geodf_driver,
            data_adapter=mock_geodataframe_adapter,
            uri="testuri",
        )
        gdf1 = data_source.read_data(bbox=list(geodf.total_bounds), buffer=1000)
        assert isinstance(gdf1, gpd.GeoDataFrame)
        assert np.all(gdf1 == geodf)

    @pytest.mark.integration()
    def test_get_data(self, geodf: gpd.GeoDataFrame, example_geojson: str):
        source = GeoDataFrameSource(
            name="test",
            uri=example_geojson,
            data_adapter=GeoDataFrameAdapter(),
            driver=PyogrioDriver(metadata_resolver=ConventionResolver()),
        )
        gdf = source.read_data(bbox=list(geodf.total_bounds))
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert np.all(gdf == geodf)

    @pytest.mark.integration()
    def test_get_data_rename(self, geodf: gpd.GeoDataFrame, example_geojson: str):
        source = GeoDataFrameSource(
            name="test",
            uri=example_geojson,
            data_adapter=GeoDataFrameAdapter(rename={"city": "ciudad"}),
            driver=PyogrioDriver(metadata_resolver=ConventionResolver()),
        )
        gdf = source.read_data(
            bbox=list(geodf.total_bounds),
            buffer=1000,
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "ciudad" in gdf.columns
        assert "city" not in gdf.columns

    @pytest.mark.integration()
    def test_get_data_not_found(self):
        source = GeoDataFrameSource(
            name="test",
            uri="no_file.geojson",
            data_adapter=GeoDataFrameAdapter(),
            driver=PyogrioDriver(metadata_resolver=ConventionResolver()),
        )
        with pytest.raises(FileNotFoundError):
            source.read_data()

    def test_instantiate_directly(
        self,
    ):
        GeoDataFrameSource(
            name="test",
            uri="points.geojson",
            driver={"name": "pyogrio", "metadata_resolver": "convention"},
            data_adapter={"unit_add": {"geoattr": 1.0}},
        )

    def test_instantiate_mixed_objects(self):
        GeoDataFrameSource(
            name="test",
            uri="points.geojson",
            driver=PyogrioDriver(
                metadata_resolver={"name": "convention", "unit_add": {"geoattr": 1.0}}
            ),
            data_adapter={"unit_add": {"geoattr": 1.0}},
        )

    def test_instantiate_directly_minimal_kwargs(self):
        GeoDataFrameSource(
            name="test",
            uri="points.geojson",
            driver={"name": "pyogrio"},
        )

    @pytest.fixture(scope="class")
    def MockDriver(self, geodf: gpd.GeoDataFrame):
        class MockGeoDataFrameDriver(GeoDataFrameDriver):
            name = "mock_geodf_to_file"

            def write(self, path: StrPath, gdf: gpd.GeoDataFrame, **kwargs) -> None:
                pass

            def read(self, uri: str, **kwargs) -> gpd.GeoDataFrame:
                return geodf

        return MockGeoDataFrameDriver

    def test_to_file(self, MockDriver: Type[GeoDataFrameSource]):
        mock_driver = MockDriver()

        source = GeoDataFrameSource(
            name="test", uri="points.geojson", driver=mock_driver
        )
        new_source = source.to_file("test")
        assert "local" in new_source.driver.filesystem.protocol
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(mock_driver) != id(new_source.driver)

    def test_to_file_override(self, MockDriver: Type[GeoDataFrameDriver]):
        driver1 = MockDriver()
        source = GeoDataFrameSource(name="test", uri="points.geojson", driver=driver1)
        driver2 = MockDriver(filesystem="memory")
        new_source = source.to_file("test", driver_override=driver2)
        assert new_source.driver.filesystem.protocol == "memory"
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(driver2) == id(new_source.driver)

    @pytest.mark.skip("Missing driver: 'raster'")
    def test_geodataframe_unit_attrs(self, artifact_data: DataCatalog):
        source = artifact_data.get_source("gadm_level1")
        source.attrs = {"NAME_0": {"long_name": "Country names"}}
        gdf = GeoDataFrameAdapter(source=source).get_data("gadm_level1")
        assert gdf["NAME_0"].attrs["long_name"] == "Country names"

        # gadm_level1 = {"gadm_level1": artifact_data.get_source("gadm_level1").to_dict()}
        # attrs = {"NAME_0": {"long_name": "Country names"}}
        # gadm_level1["gadm_level1"].update(dict(attrs=attrs))
        # artifact_data.from_dict(gadm_level1)
        # gadm_level1_gdf = artifact_data.get_geodataframe("gadm_level1")
        # assert gadm_level1_gdf["NAME_0"].attrs["long_name"] == "Country names"

    @pytest.mark.skip("Missing 'raster' driver implementation.")
    def test_to_stac_geodataframe(self, geodf: gpd.GeoDataFrame, tmp_dir: Path):
        fn_gdf = str(tmp_dir / "test.geojson")
        geodf.to_file(fn_gdf, driver="GeoJSON")
        data_catalog = DataCatalog()  # read artifacts
        _ = data_catalog.sources  # load artifact data as fallback

        # geodataframe
        name = "gadm_level1"
        adapter = cast(
            GeoDataFrameAdapter, data_catalog.get_source(name)
        )  # TODO: Fails because we have not implemented RasterDataset
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
