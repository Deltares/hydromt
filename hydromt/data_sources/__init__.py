"""DataSources responsible for validating the DataCatalog."""

from .data_source import DataSource
from .geodataframe_data_source import GeoDataFrameDataSource

__all__ = ["DataSource", "GeoDataFrameDataSource"]
