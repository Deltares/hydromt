"""Resolvers detected."""
from typing import Type

from .convention_resolver import ConventionResolver
from .metadata_resolver import MetaDataResolver

# placeholder for proper plugin behaviour later on.
RESOLVERS: dict[str, Type[MetaDataResolver]] = {
    "convention_resolver": ConventionResolver
}
