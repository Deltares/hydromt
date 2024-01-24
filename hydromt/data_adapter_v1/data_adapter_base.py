"""Base class for data adapters."""
from pydantic import BaseModel

from hydromt.data_sources.data_source import DataSource


class DataAdapterBase(BaseModel):
    """Base class for data adapters."""

    source: DataSource
