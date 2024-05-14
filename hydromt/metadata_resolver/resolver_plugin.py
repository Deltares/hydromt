"""Resolvers detected."""

from typing import Dict, Type

from .convention_resolver import ConventionResolver
from .metadata_resolver import MetaDataResolver

# placeholder for proper plugin behaviour later on.
RESOLVERS: Dict[str, Type[MetaDataResolver]] = {"convention": ConventionResolver}
