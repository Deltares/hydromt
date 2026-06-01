import warnings

from hydromt.log import *  # noqa: F403

warnings.warn(
    "importing 'log' from 'hydromt._utils' is deprecated. "
    "use 'from hydromt import log' instead.",
    DeprecationWarning,
    stacklevel=2,
)
