from packaging.version import Version

__all__ = []

HAS_XUGRID = False
HAS_PCRASTER = True  # don't check PCRASTER compat for now, see below
HAS_SHAPELY20 = False
HAS_PYGEOS = False
HAS_GCSFS = False
HAS_S3FS = False

try:
    from shapely import __version__ as SH_VERSION

    if Version(SH_VERSION) >= Version("2.0.0"):
        HAS_SHAPELY20 = True
except ImportError:
    pass

try:
    import pygeos

    HAS_PYGEOS = True
except ImportError:
    pass

# causes malloc / corrupted size errors on linux & github CI
# try:
#     import pcraster

#     HAS_PCRASTER = True
# except ImportError:
#     pass


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
