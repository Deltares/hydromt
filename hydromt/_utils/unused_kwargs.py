"""Utilities for logging unused kwargs."""

from typing import Any, Dict

from hydromt._utils.log import get_hydromt_logger

logger = get_hydromt_logger(__name__)

__all__ = ["_warn_on_unused_kwargs"]


def _warn_on_unused_kwargs(obj_name: str, name_value_dict: Dict[str, Any]):
    """Warn on unused kwargs."""
    for name, value in name_value_dict.items():
        if value is not None:
            logger.warning(
                f"object: {obj_name} does not use kwarg {name} with value {value}."
            )
