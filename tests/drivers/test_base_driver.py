from hydromt.driver.base_driver import BaseDriver
from hydromt.driver.pyogrio_driver import PyogrioDriver
from hydromt.metadata_resolver.convention_resolver import ConventionResolver


class TestBaseDriver:
    def test_init_dict_all_explicit(self):
        driver: BaseDriver = BaseDriver.model_validate(
            {"name": "pyogrio", "metadata_resolver": "convention"}
        )
        assert type(driver) == PyogrioDriver
        assert type(driver.metadata_resolver) == ConventionResolver

    def test_init_dict_minimal_args(self):
        driver: BaseDriver = BaseDriver.model_validate({"name": "pyogrio"})
        assert type(driver) == PyogrioDriver
        assert type(driver.metadata_resolver) == ConventionResolver
