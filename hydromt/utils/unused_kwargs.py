"""Utilities for logging unused kwargs."""
from functools import wraps
from logging import Logger
from typing import Any, Dict, List


def unused_kwargs_method(unused_kwargs: List[str], logger: Logger):
    """Mark unused kwargs and warn."""

    def decorator(f):
        @wraps(f)
        def inner(self: object, *args, **kwargs):
            for kwarg in unused_kwargs:
                if value := kwargs.get(kwarg) is not None:
                    logger.warning(
                        f"class: {self.__class__.__name__} does not implement kwarg "
                        f"{kwarg} with value {value}."
                    )

            return f(self, *args, **kwargs)

        return inner

    return decorator


def warn_on_unused_kwargs(
    self: object, name_value_dict: Dict[str, Any], logger: Logger
):
    """Warn on unused kwargs."""
    for name, value in name_value_dict.items():
        if value is not None:
            logger.warning(
                f"class: {self.__class__.__name__} does not implement kwarg "
                f"{name} with value {value}."
            )
