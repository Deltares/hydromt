from typing import ClassVar

from pydantic import Field

from hydromt.data_catalog.drivers.base_driver import BaseDriver, DriverOptions


class TestBaseDriver:
    def test_init_dict_all_explicit(self):
        driver: BaseDriver = BaseDriver.model_validate(
            {
                "name": "pyogrio",
                "filesystem": "memory",
            }
        )

        assert driver.__class__.__qualname__ == "PyogrioDriver"
        assert driver.filesystem.__class__.__qualname__ == "MemoryFileSystem"

    def test_init_dict_minimal_args(self):
        driver: BaseDriver = BaseDriver.model_validate({"name": "pyogrio"})

        assert driver.__class__.__qualname__ == "PyogrioDriver"
        assert driver.filesystem.__class__.__qualname__ == "LocalFileSystem"

    def test_serializes_name(self):
        driver = BaseDriver.model_validate({"name": "pyogrio"})
        assert driver.model_dump().get("name") == "pyogrio"


class TestDriverOptions:
    class CustomOptions(DriverOptions):
        """Custom options for testing."""

        KWARGS_FOR_OPEN: ClassVar[set[str]] = {"custom_kwarg1"}

        custom_kwarg1: str = Field(
            default="value1",
            description="since this is in KWARGS_FOR_OPEN, it should be returned by get_kwargs().",
        )
        custom_kwarg2: int = Field(
            default=123,
            description="since this is not in KWARGS_FOR_OPEN, it should not be returned by get_kwargs().",
        )

    def test_getKWARGS_FOR_OPEN(self):
        dct = {
            "chunks": {"x": 100, "y": 100},  # not a declared field, so always included
            "decode_times": True,  # not a declared field, so always included
            "custom_kwarg1": "value1",  # in KWARGS_FOR_OPEN, so included
            "custom_kwarg2": 123,  # not in KWARGS_FOR_OPEN, so not included
        }
        options = self.CustomOptions(**dct)
        kwargs = options.get_kwargs()
        dct.pop("custom_kwarg2")  # should not be in kwargs so remove
        assert kwargs == dct
