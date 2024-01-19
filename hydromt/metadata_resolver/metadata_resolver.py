"""Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""
from abc import ABC, abstractmethod

from hydromt.data_sources.data_source import DataSource


class MetaDataResolver(ABC):
    """Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""

    def __init__(self, source: DataSource, **kwargs):
        self._source = source
        self._kwargs = kwargs

    @property
    def source(self) -> DataSource:
        """Getter for DataSource."""
        return self.source

    @abstractmethod
    def resolve_uri(self, uri: str, **kwargs) -> list[str]:
        """Resolve metadata of data behind a single URI."""
        # TODO: add `DataAdapter.get_X` argument handling here. STAC catalog for example takes Geometry filters.
        ...
