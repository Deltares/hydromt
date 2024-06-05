"""Module for managing data catalogs and components, such as data sources, drivers, adapters and uri_resolvers."""

from .data_catalog import DataCatalog
from .predefined_catalog import PredefinedCatalog

__all__ = ["DataCatalog", "PredefinedCatalog"]
