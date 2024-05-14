"""MetaDataResolvers obtain multiple URIs before being passed to Drivers."""
from .convention_resolver import ConventionResolver
from .metadata_resolver import MetaDataResolver

__all__ = ["ConventionResolver", "MetaDataResolver"]
