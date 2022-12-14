__all__ = ["HAS_PCRASTER", "HAS_XUGRID"]

HAS_XUGRID = False
HAS_PCRASTER = False
HAS_PYGEOS = False

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
