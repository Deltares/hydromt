"""URIResolvers obtain multiple URIs before being passed to Drivers."""

from hydromt.data_catalog.uri_resolvers.convention_resolver import ConventionResolver
from hydromt.data_catalog.uri_resolvers.raster_tindex_resolver import (
    RasterTindexResolver,
)
from hydromt.data_catalog.uri_resolvers.uri_resolver import URIResolver

__all__ = [
    "ConventionResolver",
    "URIResolver",
    "RasterTindexResolver",
]

# define hydromt uri resolver entry points
# see also hydromt.uri_resolver group in pyproject.toml
__hydromt_eps__ = [
    "ConventionResolver",
    "RasterTindexResolver",
]
