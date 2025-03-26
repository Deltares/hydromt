import sys
from typing import List

from packaging.version import Version

__all__: List[str] = []

HAS_PYET = False
HAS_GCSFS = False
HAS_S3FS = False
HAS_OPENPYXL = False

try:
    import openpyxl

    HAS_OPENPYXL = True
except ImportError:
    pass

try:
    import gcsfs

    HAS_GCSFS = True

except ImportError:
    pass

try:
    import s3fs

    HAS_S3FS = True

except ImportError:
    pass

try:
    import pyet

    HAS_PYET = True

except ModuleNotFoundError:
    pass


# entrypoints in standard library only compatible from 3.10 onwards
py_version = sys.version_info
if py_version[0] >= 3 and py_version[1] >= 10:
    from importlib.metadata import entry_points, Distribution, EntryPoint, EntryPoints  # noqa: I001
else:
    from importlib_metadata import entry_points, Distribution, EntryPoint, EntryPoints  # noqa: I001
