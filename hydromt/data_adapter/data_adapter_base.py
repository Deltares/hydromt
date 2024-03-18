"""BaseModel for DataAdapter."""

from typing import Any, Dict, Union

from pydantic import BaseModel, Field


class DataAdapterBase(BaseModel):
    """BaseModel for DataAdapter."""

    unit_add: Dict[str, Any] = Field(default_factory=dict)
    unit_mult: Dict[str, Any] = Field(default_factory=dict)
    rename: Dict[str, str] = Field(default_factory=dict)
    attrs: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
    nodata: Union[float, int, Dict[str, Union[float, int]], None] = Field(default=None)
