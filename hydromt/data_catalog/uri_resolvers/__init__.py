"""MetaDataResolvers obtain multiple URIs before being passed to Drivers."""
from .convention_resolver import ConventionResolver
from .metadata_resolver import MetaDataResolver
from .raster_tindex_resolver import RasterTindexResolver

__all__ = [
    "ConventionResolver",
    "MetaDataResolver",
    "RasterTindexResolver",
]
