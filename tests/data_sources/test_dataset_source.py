from pathlib import Path
from typing import ClassVar, List, Type

import pytest
import xarray as xr
from pydantic import ValidationError
from pystac import Catalog as StacCatalog

from hydromt._typing import SourceMetadata, StrPath
from hydromt.data_catalog.adapters import DatasetAdapter
from hydromt.data_catalog.drivers import DatasetDriver
from hydromt.data_catalog.sources import DatasetSource
from hydromt.data_catalog.uri_resolvers import MetaDataResolver


@pytest.fixture(scope="session")
def mock_ds_adapter():
    class MockDatasetAdapter(DatasetAdapter):
        def transform(self, ds: xr.Dataset, metadata: SourceMetadata, **kwargs):
            return ds

    return MockDatasetAdapter()


@pytest.fixture()
def MockDatasetDriver(timeseries_ds: xr.Dataset) -> Type[DatasetDriver]:
    class MockDatasetDriver(DatasetDriver):
        name = "mock_ds_driver"
        supports_writing = True

        def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
            pass

        def read(self, uri: str, **kwargs) -> xr.Dataset:
            return self.read_data([uri], **kwargs)

        def read_data(self, uris: List[str], **kwargs) -> xr.Dataset:
            return timeseries_ds

    return MockDatasetDriver


class TestDatasetSource:
    def test_validators(self, mock_ds_adapter: DatasetAdapter):
        with pytest.raises(ValidationError) as e_info:
            DatasetSource(
                name="name",
                uri="uri",
                data_adapter=mock_ds_adapter,
                driver="does not exist",
            )

        assert e_info.value.error_count() == 1
        error_driver = next(
            filter(lambda e: e["loc"] == ("driver",), e_info.value.errors())
        )
        assert error_driver["type"] == "value_error"

    def test_model_validate(
        self,
        MockDatasetDriver: Type[DatasetDriver],
        mock_ds_adapter: DatasetAdapter,
    ):
        DatasetSource.model_validate(
            {
                "name": "example_file",
                "driver": MockDatasetDriver(),
                "data_adapter": mock_ds_adapter,
                "uri": "test_uri",
            }
        )
        with pytest.raises(ValidationError, match="'data_type' must be 'Dataset'."):
            DatasetSource.model_validate(
                {
                    "name": "geojsonfile",
                    "data_type": "DifferentDataType",
                    "driver": MockDatasetDriver(),
                    "data_adapter": mock_ds_adapter,
                    "uri": "test_uri",
                }
            )

    def test_instantiate_directly(
        self,
    ):
        datasource = DatasetSource(
            name="test",
            uri="points.nc",
            driver={"name": "dataset_xarray", "metadata_resolver": "convention"},
            data_adapter={"unit_add": {"geoattr": 1.0}},
        )
        assert isinstance(datasource, DatasetSource)

    def test_instantiate_directly_minimal_kwargs(self):
        DatasetSource(
            name="test",
            uri="points.nc",
            driver={"name": "dataset_xarray"},
        )

    def test_read_data(
        self,
        MockDatasetDriver: Type[DatasetDriver],
        mock_ds_adapter: DatasetAdapter,
        timeseries_ds: xr.Dataset,
        tmp_dir: Path,
    ):
        tmp_dir.touch("test.nc")
        source = DatasetSource(
            root=".",
            name="example_source",
            driver=MockDatasetDriver(),
            data_adapter=mock_ds_adapter,
            uri=str(tmp_dir / "test.nc"),
        )
        xr.testing.assert_equal(timeseries_ds, source.read_data())

    def test_to_file(self, MockDatasetDriver: Type[DatasetDriver]):
        mock_ds_driver = MockDatasetDriver()
        source = DatasetSource(
            name="test",
            uri="source.nc",
            driver=mock_ds_driver,
            metadata=SourceMetadata(crs=4326),
        )
        new_source = source.to_file("test.nc")
        assert "local" in new_source.driver.filesystem.protocol
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(mock_ds_driver) != id(new_source.driver)

    def test_to_file_override(self, MockDatasetDriver: Type[DatasetDriver]):
        driver1 = MockDatasetDriver()
        source = DatasetSource(
            name="test",
            uri="ds.nc",
            driver=driver1,
            metadata=SourceMetadata(category="test"),
        )
        driver2 = MockDatasetDriver(filesystem="memory")
        new_source = source.to_file("test", driver_override=driver2)
        assert new_source.driver.filesystem.protocol == "memory"
        # make sure we are not changing the state
        assert id(new_source) != id(source)
        assert id(driver2) == id(new_source.driver)

    def test_to_file_defaults(
        self, tmp_dir: Path, timeseries_ds: xr.Dataset, mock_resolver: MetaDataResolver
    ):
        class NotWritableDriver(DatasetDriver):
            name: ClassVar[str] = "test_to_file_defaults"

            def read_data(self, uri: str, **kwargs) -> xr.Dataset:
                return timeseries_ds

        old_path: Path = tmp_dir / "test.nc"
        new_path: Path = tmp_dir / "temp.nc"

        new_source: DatasetSource = DatasetSource(
            name="test",
            uri=str(old_path),
            driver=NotWritableDriver(metadata_resolver=mock_resolver),
        ).to_file(new_path)
        assert new_path.is_file()
        assert new_source.root is None
        assert new_source.driver.filesystem.protocol == ("file", "local")

    def test_dataset_to_stac_catalog(self, tmp_path: Path, timeseries_ds: xr.Dataset):
        path = tmp_path / "test.nc"
        timeseries_ds.to_netcdf(path)
        source_name = "timeseries_dataset"
        dataset_adapter = DatasetSource(
            uri=str(path), driver="dataset_xarray", name=source_name
        )

        stac_catalog = dataset_adapter.to_stac_catalog()
        assert isinstance(stac_catalog, StacCatalog)
        stac_item = next(stac_catalog.get_items(source_name), None)
        assert list(stac_item.assets.keys())[0] == "test.nc"
