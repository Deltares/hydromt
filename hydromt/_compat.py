from packaging.version import Version

__all__ = []

HAS_XUGRID = False
HAS_SHAPELY20 = False
HAS_PYET = False
HAS_GCSFS = False
HAS_S3FS = False
HAS_OPENPYXL = False
HAS_RIO_VRT = False

try:
    from shapely import __version__ as SH_VERSION

    if Version(SH_VERSION) >= Version("2.0.0"):
        HAS_SHAPELY20 = True
except ImportError:
    pass

try:
    import openpyxl

    HAS_OPENPYXL = True
except ImportError:
    pass

try:
    import xugrid

    HAS_XUGRID = True
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


try:
    import rio_vrt

    HAS_RIO_VRT = True

except ModuleNotFoundError:
    pass
