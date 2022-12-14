from packaging.version import Version
import shapely
import geopandas as gpd

__all__ = []

HAS_XUGRID = False
HAS_PCRASTER = False
HAS_PYGEOS = False

SHAPELY_GE_20 = Version(shapely.__version__) >= Version("2.0.0")

try:
    import pygeos

    if not SHAPELY_GE_20:
        gpd.options.use_pygeos = True
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
