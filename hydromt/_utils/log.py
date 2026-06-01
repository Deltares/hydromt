import warnings

from hydromt import log

warnings.warn(
    "importing 'log' from 'hydromt._utils' is deprecated. "
    "use 'from hydromt import log' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["log"]
