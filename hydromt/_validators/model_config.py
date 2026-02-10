"""Pydantic models for the validation of model config files."""

import inspect
import logging
from keyword import iskeyword
from pathlib import Path
from typing import Any, Callable, Type

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)

from hydromt.plugins import PLUGINS

logger = logging.getLogger(__name__)


class ComponentSpec(BaseModel):
    """Represents the specification for a single model component.

    The `name` field is the name of the component, which will be used to reference it in workflow steps.
    The `type` field specifies the component class, which can be given as a string referring to a registered component plugin or as the class itself.
    The remaining fields are arbitrary and will be passed as kwargs to the component constructor when the model is instantiated.
    """

    name: str
    type: Type  # perhaps rename attr to `component_type`

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


class WorkflowStep(BaseModel):
    """Represents a single step in the workflow, which will be executed as part of the model run.

    The `name` field specifies the function to call, which can be in the format `<component_name>.<function>` or just `<function>` for top-level model methods.
    The `kwargs` field contains the keyword arguments to pass to the function when executing the step.
    The `_method` field is populated at runtime when the step is bound to a model instance by calling `bind_to_model`, and is not part of the input validation.
    """

    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

    # runtime-only, set with `bind_to_model`
    _method: Callable[..., Any] | None = PrivateAttr(default=None)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def unpack(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Each step must be a dictionary")

        # already normalized shape
        if "name" in v:
            return v

        # YAML shape: exactly one key, value is kwargs
        if len(v) != 1:
            raise ValueError(
                "Each step must be a single-key dictionary or contain 'name' and 'kwargs'"
            )

        # dict with one key as name and value as kwargs
        name, kwargs = next(iter(v.items()))
        return {"name": name, "kwargs": kwargs or {}}

    @field_validator("name", mode="after")
    def validate_name(cls, name: str) -> str:
        if name.count(".") not in (0, 1):
            raise ValueError(
                f"Invalid step name '{name}': too many '.' characters. Expected format: '<component_name>.<function>' or '<function>'"
            )
        return name

    def bind_to_model(self, model_instance: type) -> None:
        """Bind this workflow step to a specific model instance.

        This is done by resolving the function to call based on self.name and the model's components.
        """
        if not hasattr(model_instance, "components"):
            raise ValueError(
                f"Model instance of type {type(model_instance).__name__} does not have 'components' attribute"
            )
        elif not isinstance(model_instance.components, dict):
            raise ValueError(
                f"Model instance 'components' attribute must be a dictionary, got {type(model_instance.components).__name__}"
            )

        comp_name, fn_name = (
            self.name.split(".") if "." in self.name else (None, self.name)
        )
        instance = model_instance.components.get(comp_name, model_instance)

        try:
            fn = getattr(instance, fn_name)
        except (AttributeError, KeyError):
            raise ValueError(
                f"Step '{self.name}' not found on {type(instance).__name__}"
            )

        if not hasattr(fn, "__ishydromtstep__"):
            raise ValueError(
                f"Step function '{self.name}' is not decorated with @hydromt_step"
            )

        # Validate signature
        sig = inspect.signature(fn)
        try:
            if "self" in sig.parameters:
                sig.bind(**{"self": None, **self.kwargs})
            else:
                sig.bind(**self.kwargs)
        except TypeError as e:
            raise ValueError(
                f"Step function '{self.name}' argument validation failed: {e}"
            )
        self._method = fn

    def execute(self):
        """Execute this workflow step by calling the bound function with the provided arguments."""
        if self._method is None:
            raise ValueError(f"Step '{self.name}' is not bound to a function")
        kwargs = self.kwargs or {}
        for k, v in kwargs.items():
            logger.info(f"{self.name}.{k}={v}")
        self._method(**kwargs)


class ModelSpec(BaseModel):
    """Represents the 'global' part of the workflow that defines the model configuration.

    The attributes of this instance will be passed as kwargs to the model constructor,
    so it is designed to be flexible and allow for any keys.
    """

    components: list[ComponentSpec] = Field(default_factory=list)
    data_libs: list[str | Path] = Field(
        default_factory=list
    )  # perhaps rename attribute to `data_catalogs`?

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


class WorkflowSpec(BaseModel):
    """Represents the entire workflow, including the model type, global model configuration and the list of workflow steps.

    The `modeltype` field specifies the model class to instantiate, which can be given as a string referring to a registered model plugin or as the class itself.
    The `globals_` field contains the model configuration that will be passed as kwargs to the model constructor when instantiating the model.
    The `steps` field is a list of workflow steps that will be executed in order as part of the model run.
    """

    # TODO: modeltype and globals_ should be converted into an instance of the model class here with a nice validator.

    modeltype: type  # perhaps rename attribute to `model_class`?
    globals_: ModelSpec = Field(
        alias="global"
    )  # perhaps rename attribute to `model_config`
    steps: list[WorkflowStep]

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


def create_workflow_step(step_dict: dict[str, dict[str, Any]]) -> WorkflowStep:
    if isinstance(step_dict, WorkflowStep):
        return step_dict
    return WorkflowStep.model_validate(step_dict)
