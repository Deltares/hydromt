import logging
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyflwdir import Flwdir, FlwdirRaster
from scipy import ndimage

from ..gis_utils import spread2d

logger = logging.Logger(__name__)

__all__ = ["river_width", "river_depth"]


def river_width(
    gdf_stream: gpd.GeoDataFrame,
    da_rivmask: xr.DataArray,
    nmin=5,
) -> np.ndarray:
    """Return segment average river width based on a river mask raster.
    For each segment in gdf_stream the associated area is calculated from stream mask
    and divided by the segment length to obtain the average width.

    Parameters
    ----------
    gdf_stream : gpd.GeoDataFrame
        River segments
    da_rivmask : xr.DataArray
        Boolean river mask in projected grid.
    nmin : int, optional
        Minimum number of cells in rivmask to calculate the width, by default 5

    Returns
    -------
    rivwth : np.ndarray
        Average width per segment in gdf_stream
    """
    assert da_rivmask.raster.crs.is_projected
    gdf_stream = gdf_stream.copy()
    # get/check river length
    if "rivlen" not in gdf_stream.columns:
        gdf_stream["rivlen"] = gdf_stream.to_crs(da_rivmask.raster.crs).length
    # rasterize streams
    gdf_stream["segid"] = np.arange(1, gdf_stream.index.size + 1, dtype=np.int32)
    segid = da_rivmask.raster.rasterize(gdf_stream, "segid").astype(np.int32)
    segid.raster.set_nodata(0)
    segid.name = "segid"
    # remove islands to get total width of braided rivers
    da_mask = da_rivmask.copy()
    da_mask.data = ndimage.binary_fill_holes(da_mask.values)
    # find nearest stream segment for all river cells
    segid_spread = spread2d(segid, da_mask)
    # get average width based on da_rivmask area and segment length
    cellarea = abs(np.multiply(*da_rivmask.raster.res))
    seg_count = ndimage.sum(
        da_rivmask, segid_spread["segid"].values, gdf_stream["segid"].values
    )
    rivwth = seg_count * cellarea / gdf_stream["rivlen"]
    valid = np.logical_and(gdf_stream["rivlen"] > 0, seg_count > nmin)
    return np.where(valid, rivwth, -9999)


def river_depth(
    data: Union[xr.Dataset, pd.DataFrame, gpd.GeoDataFrame],
    method: str,
    flwdir: Union[Flwdir, FlwdirRaster] = None,
    min_rivdph: float = 1.0,
    manning: float = 0.03,
    qbankfull_name: str = "qbankfull",
    rivwth_name: str = "rivwth",
    rivzs_name: str = "rivzs",
    rivdst_name: str = "rivdst",
    rivslp_name: str = "rivslp",
    rivman_name: str = "rivman",
    **kwargs,
) -> Union[xr.DataArray, np.ndarray]:
    """Derive river depth estimates based bankfull discharge.

    Parameters
    ----------
    data : xr.Dataset, pd.DataFrame, gpd.GeoDataFrame
        Dataset/DataFrame containing required variables
    method : {'powlaw', 'manning', 'gvf'}
        Method to estimate the river depth:

        * powlaw [1]_ [2]_: power-law hc*Qbf**hp, requires bankfull discharge (Qbf) variable in `data`.
          Optionally, `hc` (default = 0.27) and `hp` (default = 0.30) set through `kwargs`.
        * manning [3]_: river depth for kinematic conditions, requires bankfull discharge,
          river width, river slope in `data`; the river manning roughness either in data
          or as constant and optionally `min_rivslp` (default = 1e-5) set through `kwargs`.
        * gvf [4]_: gradually varying flow, requires bankfull discharge,
          river width, river surface elevation in `data`; the river manning roughness either in data
          or as constant and optionally `min_rivslp` (default = 1e-5) set through `kwargs`.
    flwdir : Flwdir, FlwdirRaster, optional
        Flow directions, required if method is not powlaw
    min_rivdph : float, optional
        Minimum river depth [m], by default 1.0
    manning : float, optional
        Constant manning roughness [s/m^{1/3}] used if `rivman_name` not in data,
        by default 0.03
    qbankfull_name, rivwth_name, rivzs_name, rivdst_name, rivslp_name, rivman_name: str, optional
        Name for variables in data: bankfull discharge [m3/s], river width [m],
        bankfull water surface elevation profile [m+REF], distance to river outlet [m],
        river slope [m/m] and river manning roughness [s/m^{1/3}]

    Returns
    -------
    rivdph: xr.DataArray, np.ndarray
        River depth [m]. A DataArray is returned if the input data is a Dataset, otherwise
        a array with the shape of one input data variable is returned.

    References
    ----------
    .. [1] Leopold & Maddock (1953). The hydraulic geometry of stream channels and some physiographic implications (No. 252; Professional Paper). U.S. Government Printing Office. https://doi.org/10.3133/pp252
    .. [2] Andreadis et al. (2013). A simple global river bankfull width and depth database. Water Resources Research, 49(10), 7164–7168. https://doi.org/10.1002/wrcr.20440
    .. [3] Sampson et al. (2015). A high-resolution global flood hazard model. Water Resources Research, 51(9), 7358–7381. https://doi.org/10.1002/2015WR016954
    .. [4] Neal et al. (2021). Estimating river channel bathymetry in large scale flood inundation models. Water Resources Research, 57(5). https://doi.org/10.1029/2020wr028301

    See Also
    --------
    pyflwdir.FlwdirRaster.river_depth
    """
    methods = ["powlaw", "manning", "gvf"]
    if method == "powlaw":

        def rivdph_powlaw(qbankfull, hc=0.27, hp=0.30, min_rivdph=1.0):
            return np.maximum(hc * qbankfull**hp, min_rivdph)

        rivdph = rivdph_powlaw(data[qbankfull_name], min_rivdph=min_rivdph, **kwargs)
    elif method in ["manning", "gvf"]:
        assert flwdir is not None
        rivdph = flwdir.river_depth(
            qbankfull=data[qbankfull_name].values,
            rivwth=data[rivwth_name].values,
            zs=data[rivzs_name].values if rivzs_name in data else None,
            rivdst=data[rivdst_name].values if rivdst_name in data else None,
            rivslp=data[rivslp_name].values if rivslp_name in data else None,
            manning=data[rivman_name].values if rivman_name in data else manning,
            method=method,
            min_rivdph=min_rivdph,
            **kwargs,
        )
    else:
        raise ValueError(f"Method unknown {method}, select from {methods}")
    if isinstance(data, xr.Dataset):
        rivdph = xr.DataArray(
            dims=data.raster.dims, coords=data.raster.coords, data=rivdph
        )
        rivdph.raster.set_nodata(-9999.0)
        rivdph.raster.set_crs(data.raster.crs)
    return rivdph
