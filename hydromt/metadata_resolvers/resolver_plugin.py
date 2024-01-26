"""Resolvers detected."""
from typing import Type

from .convention_resolver import ConventionResolver
from .metadata_resolver import MetaDataResolver

RESOLVERS: dict[str, Type[MetaDataResolver]] = {
    "convention_resolver": ConventionResolver
}
