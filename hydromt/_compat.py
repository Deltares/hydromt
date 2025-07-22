import sys
from typing import List

from packaging.version import Version

__all__: List[str] = []

HAS_GCSFS = False
HAS_GDAL = False
HAS_OPENPYXL = False
HAS_PYET = False
HAS_S3FS = False

try:
    import gcsfs

    HAS_GCSFS = True
except ImportError:
    pass

try:
    from osgeo import gdal

    HAS_GDAL = True
except ImportError:
    pass

try:
    import openpyxl

    HAS_OPENPYXL = True
except ImportError:
    pass

try:
    import pyet

    HAS_PYET = True
except ModuleNotFoundError:
    pass

try:
    import s3fs

    HAS_S3FS = True
except ImportError:
    pass

# entrypoints in standard library only compatible from 3.10 onwards
py_version = sys.version_info
if py_version[0] >= 3 and py_version[1] >= 10:
    from importlib.metadata import entry_points, Distribution, EntryPoint, EntryPoints  # noqa: I001
else:
    from importlib_metadata import entry_points, Distribution, EntryPoint, EntryPoints  # noqa: I001
