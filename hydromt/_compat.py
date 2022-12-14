__all__ = ["HAS_PCRASTER", "HAS_XUGRID", "HAS_PYET"]

HAS_XUGRID = False
HAS_PCRASTER = False
HAS_PYET = False

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
    import pyet

    HAS_PYET = True

except ModuleNotFoundError:
    pass