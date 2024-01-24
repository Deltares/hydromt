"""Resolvers detected."""
from .convention_resolver import ConventionResolver
from .metadata_resolver import MetaDataResolver

RESOLVERS: dict[str, MetaDataResolver] = {"convention_resolver": ConventionResolver()}
