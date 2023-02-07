__all__ = ["HAS_PCRASTER", "HAS_XUGRID", "HAS_GCSFS"]

HAS_XUGRID = False
HAS_PCRASTER = False

try:
    import pygeos
    import geopandas as gpd

    gpd.options.use_pygeos = True
except ImportError:
    pass

try:
    import pcraster as pcr

    HAS_PCRASTER = True

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
