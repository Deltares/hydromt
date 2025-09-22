"""Runtime Settings for HydroMT."""

from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, ValidationInfo, ValidatorFunctionWrapHandler, WrapValidator
from pydantic_settings import BaseSettings


def _validate_path(
    path: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
):
    if isinstance(path, str):
        path = Path(path)
    return handler(path, info)


Pathdantic = Annotated[Path, WrapValidator(_validate_path)]


class Settings(BaseSettings):
    """Runtime Settings for HydroMT."""

    cache_root: Pathdantic = Field(default=Path.home() / ".hydromt")


SETTINGS = Settings()
