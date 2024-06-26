from hydromt.data_catalog.drivers.base_driver import BaseDriver


class TestBaseDriver:
    def test_init_dict_all_explicit(self):
        driver: BaseDriver = BaseDriver.model_validate(
            {
                "name": "pyogrio",
                "uri_resolver": "convention",
                "filesystem": "memory",
            }
        )

        assert driver.__class__.__qualname__ == "PyogrioDriver"
        assert driver.uri_resolver.__class__.__qualname__ == "ConventionResolver"
        assert driver.filesystem.__class__.__qualname__ == "MemoryFileSystem"

    def test_init_dict_minimal_args(self):
        driver: BaseDriver = BaseDriver.model_validate({"name": "pyogrio"})

        assert driver.__class__.__qualname__ == "PyogrioDriver"
        assert driver.uri_resolver.__class__.__qualname__ == "ConventionResolver"
        assert driver.filesystem.__class__.__qualname__ == "LocalFileSystem"

    def test_serializes_name(self):
        driver: BaseDriver = BaseDriver.model_validate({"name": "pyogrio"})
        assert driver.model_dump().get("name") == "pyogrio"
