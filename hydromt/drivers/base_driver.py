"""Base class for different drivers."""
from typing import Any

from pydantic import BaseModel, field_validator

from hydromt.metadata_resolvers import MetaDataResolver
from hydromt.metadata_resolvers.resolver_plugin import RESOLVERS


class BaseDriver(BaseModel):
    """Base class for different drivers.

    Is used to implement common functionality.
    """

    metadata_resolver: MetaDataResolver

    @field_validator("metadata_resolver", mode="before")
    @classmethod
    def _validate_metadata_resolver(cls, v: Any):
        if isinstance(v, str):
            if v not in RESOLVERS:
                raise ValueError(f"unknown MetaDataResolver: '{v}'.")
            return RESOLVERS.get(v)()
        elif hasattr(v, "resolve"):  # MetaDataResolver duck-typing
            return v
        else:
            raise ValueError("metadata_resolver should be string or MetaDataResolver.")
