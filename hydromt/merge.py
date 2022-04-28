import numpy as np
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
    align=True,
    mask=None,
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
    dst_bounds: list of float
        Bounding box [xmin, ymin, xmax, ymax] of destination grid
    dst_res: float
        Resolution of destination grid
    align : bool, optional
        If True, align grid with dst_res
    mask: geopands.GeoDataFrame, optional
        Mask of destination area of interest.
        Used to determine dst_crs, dst_bounds and dst_res if missing.
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
            dmask = np.logical_and(old_nodata, ~new_nodata)
            old_data[dmask] = new_data[dmask]

    elif merge_method == "last":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            dmask = ~new_nodata
            old_data[dmask] = new_data[dmask]

    elif merge_method == "min":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            dmask = np.logical_and(~old_nodata, ~new_nodata)
            old_data[dmask] = np.minimum(old_data[dmask], new_data[dmask])
            dmask = np.logical_and(old_nodata, ~new_nodata)
            old_data[dmask] = new_data[dmask]

    elif merge_method == "max":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            dmask = np.logical_and(~old_nodata, ~new_nodata)
            old_data[dmask] = np.maximum(old_data[dmask], new_data[dmask])
            dmask = np.logical_and(old_nodata, ~new_nodata)
            old_data[dmask] = new_data[dmask]

    elif merge_method == "new":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            data_overlap = np.logical_and(~old_nodata, ~new_nodata)
            assert not np.any(data_overlap)
            mask = ~new_nodata
            old_data[mask] = new_data[mask]

    elif callable(merge_method):
        copyto = merge_method

    # resolve arguments
    if data_arrays[0].ndim != 2:
        raise ValueError("Mosaic is only implemented for 2D DataArrays.")
    if dst_crs is None and (dst_bounds is not None or dst_res is not None):
        raise ValueError("dst_bounds and dst_res not understood without dst_crs.")
    idx0 = 0
    if mask is not None and (dst_res is None or dst_crs is None):
        # select array with largest overlap of valid cells if mask
        areas = []
        for da in data_arrays:
            da_clipped = da.raster.clip_geom(geom=mask, mask=True)
            n = da_clipped.raster.mask_nodata().notnull().sum().load().item()
            areas.append(n)
        idx0 = np.argmax(areas)
        # if single layer with overlap of valid cells: return clip
        if np.sum(np.array(areas) > 0) == 1:
            return data_arrays[idx0].raster.clip_geom(mask)
    da0 = data_arrays[idx0]  # used for default dst_crs and dst_res
    # dst CRS
    if dst_crs is None:
        dst_crs = da0.raster.crs
    else:
        dst_crs = CRS.from_user_input(dst_crs)
    # dst res
    if dst_res is None:
        dst_res = da0.raster.res
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
    if dst_bounds is None and mask is not None:
        w, s, e, n = mask.to_crs(dst_crs).buffer(2 * abs(x_res)).total_bounds
        # align with da0 grid
        w0, s0, e0, n0 = da0.raster.bounds
        w = w0 + np.round((w - w0) / abs(x_res)) * abs(x_res)
        n = n0 + np.round((n - n0) / abs(y_res)) * abs(y_res)
        e = w + int(round((e - w) / abs(x_res))) * abs(x_res)
        s = n - int(round((n - s) / abs(y_res))) * abs(y_res)
        align = False  # don't align with resolution
    elif dst_bounds is None:
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
    # dst transform
    width = int(round((e - w) / abs(x_res)))
    height = int(round((n - s) / abs(y_res)))
    transform = rasterio.transform.from_bounds(*dst_bounds, width, height)
    # align with dst_res
    if align:
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
    sf_lst = []
    for i, da in enumerate(data_arrays):
        if not da.raster.aligned_grid(da_out):
            # clip with buffer
            src_bbox = da_out.raster.transform_bounds(da.raster.crs)
            da = da.raster.clip_bbox(src_bbox, buffer=10)
            if np.any([da[dim].size == 0 for dim in da.raster.dims]):
                continue  # out of bounds
            # reproject
            da = da.raster.reproject(**kwargs)
        # clip to dst_bounds
        da = da.raster.clip_bbox(dst_bounds)
        if np.any([da[dim].size == 0 for dim in da.raster.dims]):
            continue  # out of bounds
        if "source_file" in da.attrs:
            sf_lst.append(da.attrs["source_file"])
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
    da_out.attrs.update(source_file="; ".join(sf_lst))
    da_out.raster.set_crs(dst_crs)
    return da_out.raster.reset_spatial_dims_attrs()
