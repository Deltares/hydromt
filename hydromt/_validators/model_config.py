"""Pydantic models for the validation of model config files."""

import inspect
from keyword import iskeyword
from pathlib import Path
from typing import Any, Callable, Type

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from hydromt.plugins import PLUGINS


class HydromtComponentConfig(BaseModel):
    name: str
    type: Type

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        if not name.isidentifier():
            raise ValueError(f"{name} is not a valid Python identifier")
        if iskeyword(name):
            raise ValueError(f"{name} is a Python reserved keyword")
        return name

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v):
        if isinstance(v, str):
            return PLUGINS.component_plugins[v]
        return v


class RawStep(BaseModel):
    """Represents the YAML structure BEFORE runtime wiring."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def unpack(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Each step must be a single-key dictionary")
        name, args = next(iter(v.items()))
        return {"name": name, "args": args or {}}

    @field_validator("name", mode="after")
    def validate_name(cls, name: str) -> str:
        if name.count(".") not in (0, 1):
            raise ValueError(
                f"Invalid step name '{name}': too many '.' characters. Expected format: '<component_name>.<function>' or '<function>'"
            )
        return name


class HydromtModelStep(BaseModel):
    name: str
    fn: Callable[..., Any]
    args: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        str_strip_whitespace=True, arbitrary_types_allowed=True, extra="forbid"
    )

    @model_validator(mode="after")
    def validate_signature(self):
        sig = inspect.signature(self.fn)
        try:
            if "self" in sig.parameters:
                sig.bind(**{"self": None, **self.args})
            else:
                sig.bind(**self.args)
        except TypeError as e:
            raise ValueError(
                f"Step function '{self.name}' argument validation failed: {e}"
            )
        return self


class HydromtGlobalConfig(BaseModel):
    components: list[HydromtComponentConfig] = Field(default_factory=list)
    data_libs: list[str | Path] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("components", mode="before")
    @classmethod
    def normalize_components(cls, v):
        if isinstance(v, dict):
            return [{"name": k, **opts} for k, opts in v.items()]
        return v

    @field_validator("data_libs", mode="before")
    @classmethod
    def resolve_paths_or_catalogs(cls, v, info):
        if not v:
            return v

        root: Path | None = info.context.get("root") if info.context else None
        resolved = []

        for entry in v:
            if entry in PLUGINS.catalog_plugins:
                resolved.append(entry)  # keep catalog names as-is
            else:
                p = Path(entry)
                if not p.is_absolute() and root:
                    p = (root / p).resolve()
                resolved.append(p)
        return resolved

    @field_serializer("components", mode="plain")
    def serialize_components_field(self, v, info):
        """Serialize components as a name-based dictionary."""
        return {
            c.name: {k: val for k, val in c.model_dump().items() if k != "name"}
            for c in v
        }


class HydromtModelSetup(BaseModel):
    modeltype: type
    globals_: HydromtGlobalConfig = Field(alias="global")
    steps: list[RawStep]

    model_config = ConfigDict(
        populate_by_name=True,  # allow "global" as field name
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @field_validator("modeltype", mode="before")
    @classmethod
    def validate_model_type(cls, v):
        if isinstance(v, str):
            if v not in PLUGINS.model_plugins:
                raise ValueError(f"Unknown model '{v}'")
            return PLUGINS.model_plugins[v]
        return v
