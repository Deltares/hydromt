"""Module for managing data catalogs and components, such as data sources, drivers, adapters and uri_resolvers."""

from hydromt.data_catalog.data_catalog import DataCatalog
from hydromt.data_catalog.predefined_catalog import PredefinedCatalog

__all__ = ["DataCatalog", "PredefinedCatalog"]
