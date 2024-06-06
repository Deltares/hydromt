import numpy as np
from numpy.typing import NDArray

__all__ = ["_elevation2rgba", "_rgba2elevation"]


def _elevation2rgba(val, nodata=np.nan) -> NDArray[np.uint8]:
    """Convert elevation to rgb tuple."""
    val += 32768
    r = np.floor(val / 256).astype(np.uint8)
    g = np.floor(val % 256).astype(np.uint8)
    b = np.floor((val - np.floor(val)) * 256).astype(np.uint8)
    mask = np.isnan(val) if np.isnan(nodata) else val == nodata
    a = np.where(mask, 0, 255).astype(np.uint8)
    return np.stack((r, g, b, a), axis=2)


def _rgba2elevation(
    rgba: np.ndarray, nodata=np.nan, dtype=np.float32
) -> NDArray[np.float32]:
    """Convert rgb tuple to elevation."""
    r, g, b, a = np.split(rgba, 4, axis=2)
    val = (r * 256 + g + b / 256) - 32768
    return np.where(a == 0, nodata, val).squeeze().astype(dtype)
