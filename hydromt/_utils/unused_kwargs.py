"""Utilities for logging unused kwargs."""

from logging import Logger, getLogger
from typing import Any, Dict

logger: Logger = getLogger(__name__)

__all__ = ["_warn_on_unused_kwargs"]


def _warn_on_unused_kwargs(obj_name: str, name_value_dict: Dict[str, Any]):
    """Warn on unused kwargs."""
    for name, value in name_value_dict.items():
        if value is not None:
            logger.warning(
                f"object: {obj_name} does not use kwarg {name} with value {value}."
            )
