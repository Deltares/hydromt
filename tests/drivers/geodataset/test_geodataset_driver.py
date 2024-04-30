from typing import Type

import pytest

from hydromt.drivers import RasterDatasetDriver


class TestRasterDatasetDriver:
    @pytest.fixture(scope="class")
    def CustomDriver(self) -> Type[RasterDatasetDriver]:
        class NewlyImplementedDriver(RasterDatasetDriver):
            name = "test_base_driver_options"

            def read_data(self, *args, **kwargs):
                return f"{self.options['prefix']}_data"

        return NewlyImplementedDriver

    def test_options(self, CustomDriver: Type[RasterDatasetDriver]):
        driver = CustomDriver.model_validate({"options": {"prefix": "my"}})
        assert "my_data" == driver.read_data()

    def test_options_overrides(self, CustomDriver: Type[RasterDatasetDriver]):
        driver = CustomDriver.model_validate({"options": {"prefix": "your"}})
        assert "your_data" == driver.read_data(prefix="your")
