"""URIResolvers obtain multiple URIs before being passed to Drivers."""

from .convention_resolver import ConventionResolver
from .raster_tindex_resolver import RasterTindexResolver
from .uri_resolver import URIResolver

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
