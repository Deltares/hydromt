"""MetaDataResolvers obtain multiple URIs before being passed to Drivers."""
from .convention_resolver import ConventionResolver

RESOLVERS: dict[str, str] = {"convention_resolver": ConventionResolver}
