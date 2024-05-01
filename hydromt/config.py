"""Runtime Settings for HydroMT."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from hydromt._typing import Pathdantic


class Settings(BaseSettings):
    """Runtime Settings for HydroMT."""

    cache_root: Pathdantic = Field(default=Path.home() / ".hydromt")


SETTINGS = Settings()
