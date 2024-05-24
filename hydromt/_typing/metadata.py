"""Metadata on DataSource."""

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing_extensions import Annotated

# always stringify version
Version = Annotated[str, BeforeValidator(str)]


class SourceMetadata(BaseModel):
    """
    Metadata for data source.

    This refers to data that is used to enrich the data format the source is in.
    SourceMetaData is not used to reproject or fill nodata values, it is used to
    check the data and enrich the metadata for HydroMT.


    Only the fields listed here are used in HydroMT, the rest are free for used
    to fill in.
    """

    model_config = ConfigDict(extra="allow")

    crs: Union[int, str, None] = None
    unit: Optional[str] = None
    extent: Dict[str, Any] = Field(default_factory=dict)
    nodata: Union[dict, float, int, None] = None
    attrs: Dict[str, Any] = Field(default_factory=dict)
    category: Optional[str] = None
