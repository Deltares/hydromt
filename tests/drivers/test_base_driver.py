from hydromt.driver.base_driver import BaseDriver


class TestBaseDriver:
    def test_init_dict_all_explicit(self):
        driver: BaseDriver = BaseDriver.model_validate(
            {
                "name": "pyogrio",
                "metadata_resolver": "convention",
                "filesystem": "memory",
            }
        )

        assert driver.__class__.__qualname__ == "PyogrioDriver"
        assert driver.metadata_resolver.__class__.__qualname__ == "ConventionResolver"
        assert driver.filesystem.__class__.__qualname__ == "MemoryFileSystem"

    def test_init_dict_minimal_args(self):
        driver: BaseDriver = BaseDriver.model_validate({"name": "pyogrio"})

        assert driver.__class__.__qualname__ == "PyogrioDriver"
        assert driver.metadata_resolver.__class__.__qualname__ == "ConventionResolver"
        assert driver.filesystem.__class__.__qualname__ == "LocalFileSystem"
