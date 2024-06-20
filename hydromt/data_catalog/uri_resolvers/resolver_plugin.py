"""Resolvers detected."""

from typing import Dict, Type

from .convention_resolver import ConventionResolver
from .metadata_resolver import MetaDataResolver
from .raster_tindex_resolver import RasterTindexResolver

# placeholder for proper plugin behaviour later on.
RESOLVERS: Dict[str, Type[MetaDataResolver]] = {
    ConventionResolver.name: ConventionResolver,
    RasterTindexResolver.name: RasterTindexResolver,
}
