from functools import reduce

__all__ = ["_rgetattr"]


def _rgetattr(obj, attr, *args):
    """Recursive get attribute from object."""

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))
