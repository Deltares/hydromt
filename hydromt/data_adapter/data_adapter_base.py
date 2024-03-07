"""BaseModel for DataAdapter."""
from pydantic import BaseModel, Field

from hydromt.data_adapter.harmonization_settings import HarmonizationSettings


class DataAdapterBase(BaseModel):
    """BaseModel for DataAdapter."""

    harmonization_settings: HarmonizationSettings = Field(
        default_factory=HarmonizationSettings
    )
