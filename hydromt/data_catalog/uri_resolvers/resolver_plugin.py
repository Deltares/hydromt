"""Resolvers detected."""

from typing import Dict, Type

from .convention_resolver import ConventionResolver
from .raster_tindex_resolver import RasterTindexResolver
from .uri_resolver import URIResolver

# placeholder for proper plugin behaviour later on.
RESOLVERS: Dict[str, Type[URIResolver]] = {
    ConventionResolver.name: ConventionResolver,
    RasterTindexResolver.name: RasterTindexResolver,
}
