"""Base class for different drivers."""

from abc import ABC
from typing import Any, Callable, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hydromt.metadata_resolver import MetaDataResolver
from hydromt.metadata_resolver.resolver_plugin import RESOLVERS


class BaseDriver(BaseModel, ABC):
    """Base class for different drivers.

    Is used to implement common functionality.
    """

    name: ClassVar[str]
    metadata_resolver: MetaDataResolver = Field(
        default_factory=RESOLVERS.get("convention")
    )
    config: ConfigDict = ConfigDict(extra="allow")

    @field_validator("metadata_resolver", mode="before")
    @classmethod
    def _validate_metadata_resolver(cls, v: Any):
        if isinstance(v, str):
            if v not in RESOLVERS:
                raise ValueError(f"unknown MetaDataResolver: '{v}'.")
            return RESOLVERS.get(v)()
        elif isinstance(v, dict):
            try:
                name: str = v.pop("name")
                if name not in RESOLVERS:
                    raise ValueError(f"unknown MetaDataResolver: '{name}'.")
                else:
                    return RESOLVERS.get(name).model_validate(v)
            except KeyError:
                # return default when name is missing
                return cls.model_fields.get("metadata_resolver").default_factory(**v)

        elif v is None:  # let default factory handle it
            return None
        elif hasattr(v, "resolve"):  # MetaDataResolver duck-typing
            return v
        else:
            raise ValueError(
                "metadata_resolver should be string, dict or MetaDataResolver."
            )

    @model_validator(mode="wrap")
    @classmethod
    def _init_driver(cls, data: Any, handler: Callable):
        """Initialize the subclass based on the 'name' class variable.

        All DataSources should be parsed based on their `name` class variable;
        e.g. a dict with `name` = `pyogrio` should be parsed as a `PyogrioDriver`.
        This class searches all subclasses until the correct driver is found and
        initialized that one.
        This allows an API as: `BaseDriver.model_validate(pyogrio_driver_dict)` or
        `BaseDriver(pyogrio_driver_dict)`, but also `PyogrioDriver(pyogrio_driver_dict)`.

        Inspired by: https://github.com/pydantic/pydantic/discussions/7008#discussioncomment
        """
        if not isinstance(data, dict):
            # Other objects should already be the correct subclass.
            return handler(data)

        if ABC not in cls.__bases__:
            # If cls is concrete, just validate as normal
            return handler(data)

        if name := data.get("name"):
            try:
                # Find which DataSource to instantiate.
                target_cls: BaseDriver = next(
                    filter(lambda sc: sc.name == name, cls._find_all_possible_types())
                )  # subclasses should be loaded from __init__.py
                return target_cls.model_validate(data)
            except StopIteration:
                raise ValueError(f"Unknown 'name': '{name}'")

        raise ValueError(f"{cls.__name__} needs 'name'")

    @classmethod
    def _find_all_possible_types(cls):
        """Recursively generate all possible types for this object."""
        # any concrete class is a possible type
        if ABC not in cls.__bases__:
            yield cls

        # continue looking for possible types in subclasses
        for subclass in cls.__subclasses__():
            yield from subclass._find_all_possible_types()
