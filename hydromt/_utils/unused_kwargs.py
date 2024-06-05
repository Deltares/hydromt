"""Utilities for logging unused kwargs."""
from logging import Logger
from typing import Any, Dict


def _warn_on_unused_kwargs(
    obj_name: str, name_value_dict: Dict[str, Any], logger: Logger
):
    """Warn on unused kwargs."""
    for name, value in name_value_dict.items():
        if value is not None:
            logger.warning(
                f"object: {obj_name} does not use kwarg " f"{name} with value {value}."
            )
