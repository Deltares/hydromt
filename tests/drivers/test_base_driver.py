from hydromt.drivers.base_driver import BaseDriver


class TestBaseDriver:
    def test_init_dict_all_explicit(self):
        driver: BaseDriver = BaseDriver.model_validate(
            {"name": "pyogrio", "metadata_resolver": "convention"}
        )

        assert driver.__class__.__qualname__ == "PyogrioDriver"
        assert driver.metadata_resolver.__class__.__qualname__ == "ConventionResolver"

    def test_init_dict_minimal_args(self):
        driver: BaseDriver = BaseDriver.model_validate({"name": "pyogrio"})

        assert driver.__class__.__qualname__ == "PyogrioDriver"
        assert driver.metadata_resolver.__class__.__qualname__ == "ConventionResolver"
