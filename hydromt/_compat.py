from packaging.version import Version

__all__ = []

HAS_XUGRID = False
HAS_PCRASTER = False
HAS_SHAPELY20 = False
HAS_PYGEOS = False

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

try:
    import pcraster

    HAS_PCRASTER = True
except ImportError:
    pass


try:
    import xugrid

    HAS_XUGRID = True
except ImportError:
    pass
