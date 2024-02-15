"""DataSources responsible for validating the DataCatalog."""

from .data_source import DataSource
from .geodataframe import GeoDataFrameDataSource

__all__ = ["DataSource", "GeoDataFrameDataSource"]
