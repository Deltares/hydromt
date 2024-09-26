"""Decorator for hydromt steps."""

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def hydromt_step(funcobj: F) -> F:
    """Decorate a method indicating it is a hydromt step.

    Only methods decorated with this decorator are allowed to be called by Model.build and Model.update.
    """
    funcobj.__ishydromtstep__ = True  # type: ignore[attr-defined]
    return funcobj
