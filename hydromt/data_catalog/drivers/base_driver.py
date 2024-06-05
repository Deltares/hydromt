"""Base class for different drivers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generator, List, Type

from fsspec.implementations.local import LocalFileSystem
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from hydromt._typing import FS
from hydromt.data_catalog.uri_resolvers import MetaDataResolver
from hydromt.data_catalog.uri_resolvers.resolver_plugin import RESOLVERS
from hydromt.plugins import PLUGINS


class BaseDriver(BaseModel, ABC):
    """Base class for different drivers.

    Is used to implement common functionality.
    """

    name: ClassVar[str]
    supports_writing: ClassVar[bool] = False
    metadata_resolver: MetaDataResolver = Field(default_factory=RESOLVERS["convention"])
    filesystem: FS = Field(default=LocalFileSystem())
    options: Dict[str, Any] = Field(default_factory=dict)

    model_config: ConfigDict = ConfigDict(extra="forbid")

    @field_validator("metadata_resolver", mode="before")
    @classmethod
    def _validate_metadata_resolver(cls, v: Any):
        if isinstance(v, str):
            if v not in RESOLVERS:
                raise ValueError(f"unknown MetaDataResolver: '{v}'.")
            return RESOLVERS[v]()
        elif isinstance(v, dict):
            try:
                name: str = v.pop("name")
                if name not in RESOLVERS:
                    raise ValueError(f"unknown MetaDataResolver: '{name}'.")
                return RESOLVERS[name].model_validate(v)
            except KeyError:
                # return default when name is missing
                return cls.model_fields["metadata_resolver"].default_factory(**v)

        elif v is None:  # let default factory handle it
            return None
        elif hasattr(
            v, MetaDataResolver.resolve.__name__
        ):  # MetaDataResolver duck-typing
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
        if isinstance(data, str):
            # name is enough for a default driver
            data = {"name": data}
        if not isinstance(data, dict):
            # Other objects should already be the correct subclass.
            return handler(data)

        if ABC not in cls.__bases__:
            # If cls is concrete, just validate as normal
            return handler(data)

        if name := data.pop("name", None):
            # Load plugins, importing subclasses of BaseDriver
            PLUGINS.driver_plugins  # noqa: B018

            # Find which Driver to instantiate.
            possible_drivers: List[Type["BaseDriver"]] = list(
                filter(lambda dr: dr.name == name, cls._find_all_possible_types())
            )
            if len(possible_drivers) == 0:
                raise ValueError(f"Unknown 'name': '{name}'")
            elif len(possible_drivers) > 1:
                raise ValueError(
                    f"""Duplication between driver name {name} in classes:
                    {list(map(lambda dr: dr.__qualname__, possible_drivers))}"""
                )
            else:
                return possible_drivers[0].model_validate(data)
        raise ValueError(f"{cls.__name__} needs 'name'")

    @classmethod
    def _find_all_possible_types(cls) -> Generator[None, None, Type["BaseDriver"]]:
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

    @model_serializer(mode="wrap")
    def _serialize(
        self,
        nxt: SerializerFunctionWrapHandler,
    ) -> Dict[str, Any]:
        """Add name to serialized result."""
        serialized: Dict[str, Any] = nxt(self)
        serialized["name"] = self.name
        return serialized

    # Args and kwargs will be refined by HydroMT subclasses.
    @abstractmethod
    def read(self, uri: str, *args, **kwargs):
        """
        Discover and read in data.

        args:
            uri: str identifying the data source.
        """

    def __eq__(self, value: object) -> bool:
        """Compare equality.

        Overwritten as filesystem between two sources can not be the same.
        """
        if isinstance(value, self.__class__):
            return self.model_dump(round_trip=True) == value.model_dump(round_trip=True)
        return super().__eq__(value)
