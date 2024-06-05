"""Decorator for hydromt steps."""


def hydromt_step(funcobj):
    """Decorate a method indicating it is a hydromt step.

    Only methods decorated with this decorator are allowed to be called by Model.build and Model.update.
    """
    funcobj.__ishydromtstep__ = True
    return funcobj
