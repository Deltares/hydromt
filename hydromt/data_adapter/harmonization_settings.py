"""Settings used to harmonize data."""
from typing import Any, Dict, Union

from pydantic import BaseModel, ConfigDict, Field


class HarmonizationSettings(BaseModel):
    """Settings used to harmonize data."""

    model_config = ConfigDict(frozen=True)

    unit_add: Dict[str, Any] = Field(default_factory=dict)
    unit_mult: Dict[str, Any] = Field(default_factory=dict)
    model_add: Dict[str, str] = Field(default_factory=dict)
    rename: Dict[str, str] = Field(default_factory=dict)
    extent: Dict[str, Any] = Field(default_factory=dict)
    attrs: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
    nodata: Union[float, int, Dict[str, Union[float, int]], None] = Field(default=None)
