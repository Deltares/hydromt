from pathlib import Path
from typing import Optional, Type

import pytest
import xarray as xr
from pystac import Catalog as StacCatalog

from hydromt._typing import ErrorHandleMethod
from hydromt.data_catalog.adapters import DatasetAdapter
from hydromt.data_catalog.drivers import DatasetDriver
from hydromt.data_catalog.sources import DatasetSource
from hydromt.data_catalog.uri_resolvers import URIResolver


class TestDatasetSource:
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

    def test_to_stac_catalog(self, tmp_path: Path, timeseries_ds: xr.Dataset):
        path = tmp_path / "test.nc"
        timeseries_ds.to_netcdf(path)
        source_name = "timeseries_dataset"
        dataset_source = DatasetSource(
            uri=str(path), driver="dataset_xarray", name=source_name
        )

        stac_catalog = dataset_source.to_stac_catalog()
        assert isinstance(stac_catalog, StacCatalog)
        stac_item = next(stac_catalog.get_items(source_name), None)
        assert list(stac_item.assets.keys())[0] == "test.nc"

    def test_to_stac_catalog_zarr(
        self, mock_resolver: URIResolver, MockDatasetDriver: Type[DatasetDriver]
    ):
        source_name = "timeseries_dataset_zarr"
        driver = MockDatasetDriver(uri_resolver=mock_resolver)
        with pytest.raises(ValueError, match="does not support zarr"):
            DatasetSource(
                uri="test.zarr",
                driver=driver,
                name=source_name,
            ).to_stac_catalog()

    @pytest.fixture()
    def dataset_source_no_timerange(
        self,
        mock_resolver: URIResolver,
        MockDatasetDriver: Type[DatasetDriver],
        monkeypatch: pytest.MonkeyPatch,
    ) -> DatasetSource:
        def _get_time_range(self, *args, **kwargs):
            raise IndexError("no timerange found.")

        source_name = "timeseries_dataset_nc"
        driver = MockDatasetDriver(uri_resolver=mock_resolver)
        monkeypatch.setattr(DatasetSource, name="get_time_range", value=_get_time_range)
        return DatasetSource(
            uri="test.nc",
            driver=driver,
            name=source_name,
        )

    def test_to_stac_catalog_skip(self, dataset_source_no_timerange: DatasetSource):
        catalog: Optional[StacCatalog] = dataset_source_no_timerange.to_stac_catalog(
            on_error=ErrorHandleMethod.SKIP
        )
        assert catalog is None

    def test_to_stac_catalog_coerce(self, dataset_source_no_timerange: DatasetSource):
        catalog: Optional[StacCatalog] = dataset_source_no_timerange.to_stac_catalog(
            on_error=ErrorHandleMethod.COERCE
        )
        assert isinstance(catalog, StacCatalog)
        stac_item = next(catalog.get_items(dataset_source_no_timerange.name), None)
        assert list(stac_item.assets.keys())[0] == "test.nc"
        assert stac_item.properties["start_datetime"] == "0001-01-01T00:00:00Z"
        assert stac_item.properties["end_datetime"] == "0001-01-01T00:00:00Z"
