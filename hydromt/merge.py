import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box

from .raster import full_from_transform


def merge(
    data_arrays,
    dst_crs=None,
    dst_bounds=None,
    dst_res=None,
    merge_method="first",
    **kwargs,
):
    """Merge multiple tiles to a single DataArray, if mismatching grid CRS or resolution,
    tiles are reprojected to match the output DataArray grid. If no destination
    grid is defined it is based on the first DataArray in the list.

    Based on :py:meth:`rasterio.merge.merge`.

    Arguments
    ----------
    data_arrays: list of xarray.DataArray
        Tiles to merge
    dst_crs: pyproj.CRS, int
        CRS (or EPSG code) of destination grid
    dst_bound: list of float
        Bounding box [xmin, ymin, xmax, ymax] of destination grid
    dst_res: float
        Resolution of destination grid
    merge_method: {'first','last','min','max','new'}, callable
        Merge method:

        * first: reverse painting
        * last: paint valid new on top of existing
        * min: pixel-wise min of existing and new
        * max: pixel-wise max of existing and new
        * new: assert no pixel overlap
    **kwargs:
        Key-word arguments passed to :py:meth:`~hydromt.raster.RasterDataArray.reproject`

    Returns
    -------
    da_out: xarray.DataArray
        Merged tiles.
    """

    # define merge method
    if merge_method == "first":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            mask = np.logical_and(old_nodata, ~new_nodata)
            old_data[mask] = new_data[mask]

    elif merge_method == "last":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            mask = ~new_nodata
            old_data[mask] = new_data[mask]

    elif merge_method == "min":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            mask = np.logical_and(~old_nodata, ~new_nodata)
            old_data[mask] = np.minimum(old_data[mask], new_data[mask])
            mask = np.logical_and(old_nodata, ~new_nodata)
            old_data[mask] = new_data[mask]

    elif merge_method == "max":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            mask = np.logical_and(~old_nodata, ~new_nodata)
            old_data[mask] = np.maximum(old_data[mask], new_data[mask])
            mask = np.logical_and(old_nodata, ~new_nodata)
            old_data[mask] = new_data[mask]

    elif merge_method == "new":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            data_overlap = np.logical_and(~old_nodata, ~new_nodata)
            assert not np.any(data_overlap)
            mask = ~new_nodata
            old_data[mask] = new_data[mask]

    elif callable(merge_method):
        copyto = merge_method

    # resolve arguments
    da0 = data_arrays[0]
    if da0.ndim != 2:
        raise ValueError("Mosaic is only implemented for 2D DataArrays.")
    # dst CRS
    if dst_crs is None:
        dst_crs = da0.raster.crs
        if dst_res is None:
            dst_res = da0.raster.res
    else:
        dst_crs = CRS.from_user_input(dst_crs)
        # rough estimate of dst_res based input rasters
        if dst_res is None:
            xs, ys = [], []
            for da in data_arrays:
                w, h, bounds = (
                    da.raster.width,
                    da.raster.height,
                    da.raster.internal_bounds,
                )
                transform0 = rasterio.warp.calculate_default_transform(
                    da.raster.crs, dst_crs, w, h, *bounds
                )[0]
                xs.append(transform0[0])
                ys.append(transform0[4])
            dst_res = np.mean(xs), np.mean(ys)
    # dst res
    if isinstance(dst_res, float):
        x_res, y_res = dst_res, -dst_res
    elif isinstance(dst_res, (tuple, list)) and len(dst_res) == 2:
        x_res, y_res = dst_res
    else:
        raise ValueError("dst_res not understood.")
    assert x_res > 0
    # TODO test y_res > 0
    dst_res = (x_res, -y_res)  # NOTE: y_res is multiplied with -1 in rasterio!
    # dst bounds
    if dst_bounds is None:
        for i, da in enumerate(data_arrays):
            if i == 0:
                w, s, e, n = da.raster.transform_bounds(dst_crs)
            else:
                w1, s1, e1, n1 = da.raster.transform_bounds(dst_crs)
                w = min(w, w1)
                s = min(s, s1)
                e = max(e, e1)
                n = max(n, n1)
    else:
        w, s, e, n = dst_bounds
    # check N>S orientation
    top, bottom = (n, s) if y_res < 0 else (s, n)
    dst_bounds = w, bottom, e, top
    # save dst geometry for clipping later
    dst_geom = gpd.GeoDataFrame(geometry=[box(*dst_bounds)], crs=dst_crs)
    # dst transform
    width = int(round((e - w) / abs(x_res)))
    height = int(round((n - s) / abs(y_res)))
    transform = rasterio.transform.from_bounds(*dst_bounds, width, height)
    # align with dst_res
    transform, width, height = rasterio.warp.aligned_target(
        transform, width, height, dst_res
    )

    # creat output array
    nodata = da0.raster.nodata
    isnan = np.isnan(nodata)
    dtype = da0.dtype
    da_out = full_from_transform(
        transform=transform,
        shape=(height, width),
        nodata=nodata,
        dtype=dtype,
        name=da0.name,
        attrs=da0.attrs,
        crs=dst_crs,
    )
    ys = da_out.raster.ycoords.values
    xs = da_out.raster.xcoords.values
    dest = da_out.values
    kwargs.update(dst_crs=dst_crs, dst_res=dst_res, align=True)
    for i, da in enumerate(data_arrays):
        # reproject
        if not da.raster.aligned_grid(da_out):
            da = da.raster.clip_geom(dst_geom, buffer=10).raster.reproject(**kwargs)
        # clip to bounds
        da = da.raster.clip_bbox(dst_bounds)
        if np.any([da[dim].size == 0 for dim in da.raster.dims]):
            continue  # out of bounds
        # merge overlap
        w0, s0, e0, n0 = da.raster.bounds
        if y_res < 0:
            top = np.where(ys <= n0)[0][0] if n0 < ys[0] else 0
            bottom = np.where(ys < s0)[0][0] if s0 > ys[-1] else None
        else:
            top = np.where(ys > n0)[0][0] if n0 < ys[-1] else 0
            bottom = np.where(ys >= s0)[0][0] if s0 > ys[0] else None
        left = np.where(xs >= w0)[0][0] if w0 > xs[0] else 0
        right = np.where(xs > e0)[0][0] if e0 < xs[-1] else None
        y_slice = slice(top, bottom)
        x_slice = slice(left, right)
        region = dest[y_slice, x_slice]
        temp = np.ma.masked_equal(da.values.astype(dtype), nodata)
        if isnan:
            region_nodata = np.isnan(region)
            temp_nodata = np.isnan(temp)
        else:
            region_nodata = region == nodata
            temp_nodata = temp.mask
        copyto(region, temp, region_nodata, temp_nodata)

    # set attrs
    da_out.raster.set_crs(dst_crs)
    return da_out.raster.reset_spatial_dims_attrs()
