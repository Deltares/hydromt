import shapely
import geopandas as gpd

__all__ = []

HAS_XUGRID = False
HAS_PCRASTER = False

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
