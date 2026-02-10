"""Parsing functions for HydroMT. Should take in data as dict and return validated model setup."""

from pathlib import Path
from typing import Any

from hydromt._utils import deep_merge
from hydromt._validators.model_config import (
    HydromtGlobalConfig,
    HydromtModelSetup,
    RawStep,
)
from hydromt.plugins import PLUGINS


def parse_workflow(
    data: dict[str, Any],
    modeltype: str | None = None,
    defaults: dict[str, Any] | None = None,
    abs_path: bool = True,
    skip_abspath_sections: list[str] | None = None,
    root: Path | None = None,
) -> HydromtModelSetup:
    """Read and validate HydroMT workflow YAML data.

    Parameters
    ----------
    data : dict[str, Any]
        Data dictionary representing the HydroMT workflow YAML file.
    modeltype : str | None, optional
        Model type to use. If None, the model type is read from the YAML file.
    defaults : dict[str, Any] | None, optional
        Default configuration values to merge with the YAML file.
    abs_path : bool, default=True
        Whether to resolve relative paths to absolute paths.
    skip_abspath_sections : list[str] | None, optional
        List of sections to skip when resolving absolute paths.
    root : Path, optional
        Root path for resolving relative paths, by default None

    Returns
    -------
    HydromtModelSetup
        The validated HydroMT model setup.
    """
    # Merge defaults
    if defaults:
        data = deep_merge(defaults, data)

    # Determine model type
    resolved_modeltype = modeltype or data.get("modeltype")
    if resolved_modeltype is None:
        raise ValueError("Model type not specified in workflow or arguments")

    # Context for path resolution
    context = {}
    if (
        abs_path
        and root is not None
        and (skip_abspath_sections is None or "global" not in skip_abspath_sections)
    ):
        context["root"] = root

    # Validate
    global_cfg = HydromtGlobalConfig.model_validate(
        data.get("global", {}), context=context
    )
    raw_steps = [
        RawStep.model_validate(s, context=context) for s in data.get("steps", [])
    ]

    return HydromtModelSetup(
        modeltype=PLUGINS.model_plugins.get(resolved_modeltype, resolved_modeltype),
        globals_=global_cfg,
        steps=raw_steps,
    )
