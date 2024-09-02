"""Abstract base class for pydantic abstract classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generator, List

from pydantic import (
    BaseModel,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)


class AbstractBaseModel(BaseModel, ABC):
    """BaseModel for Abstract pydantic models.

    Contains logic to find all non-ABC superclasses
    """

    name: ClassVar[str]

    @model_validator(mode="wrap")
    @classmethod
    def _init_subclass(cls, data: Any, handler: Callable):
        """Initialize the subclass based on the 'name' class variable.

        All subclasses should be parsed based on their `name` class variable;
        e.g. a dict with `name` = `myclass` should be parsed as a `MyClass` if
        `MyClass.name == myclass"`.
        This class searches all subclasses until the correct driver is found and
        initialized that one.
        This allows an API as: `AbstractBaseModel.model_validate(dict)` or
        `AbstractBaseModel(dict)`, but also `AbstractBaseModel(subclassdict)`.

        Inspired by: https://github.com/pydantic/pydantic/discussions/7008#discussioncomment
        """
        if isinstance(data, str):
            # name is enough for a default subclass
            data = {"name": data}
        if not isinstance(data, dict):
            # Other objects should already be the correct subclass.
            return handler(data)

        if ABC not in cls.__bases__:
            # If cls is concrete, just validate as normal
            return handler(data)

        if name := data.pop("name", None):
            # Load plugins, importing subclasses
            cls._load_plugins()

            # Find which subclass to instantiate.
            possible_subclasses: List[AbstractBaseModel] = list(
                filter(lambda dr: dr.name == name, cls._find_all_possible_types())
            )
            if len(possible_subclasses) == 0:
                raise ValueError(f"Unknown 'name': '{name}'")
            elif len(possible_subclasses) > 1:
                raise ValueError(
                    f"""Duplication between base name name {name} in subclasses of {cls}:
                    {list(map(lambda dr: dr.__qualname__, possible_subclasses))}"""
                )
            else:
                return possible_subclasses[0].model_validate(data)
        raise ValueError(f"{cls.__name__} needs 'name'")

    @model_serializer(mode="wrap")
    def _serialize(
        self,
        nxt: SerializerFunctionWrapHandler,
    ) -> Dict[str, Any]:
        """Add name to serialized result."""
        serialized: Dict[str, Any] = nxt(self)
        serialized["name"] = self.name
        return serialized

    @classmethod
    @abstractmethod
    def _load_plugins(cls):
        """Load plugins for this model."""
        ...

    @classmethod
    def _find_all_possible_types(cls) -> Generator[None, None, AbstractBaseModel]:
        """Recursively generate all possible types for this object.

        Logic relies on __bases__ and __subclass__() of the BaseDriver class,
        which means that all drivers and plugins should be loaded in before.
        """
        # any concrete class is a possible type
        if ABC not in cls.__bases__:
            yield cls

        # continue looking for possible types in subclasses
        for subclass in cls.__subclasses__():
            yield from subclass._find_all_possible_types()
