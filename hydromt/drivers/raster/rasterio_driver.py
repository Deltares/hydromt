"""Driver using rasterio for RasterDataset."""
from glob import glob
from io import IOBase
from logging import Logger, getLogger
from os.path import basename
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import numpy as np
import rasterio
import rioxarray
import xarray as xr

from hydromt._typing import (
    Geom,
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
    ZoomLevel,
)
from hydromt._typing.error import NoDataStrategy
from hydromt._utils.unused_kwargs import warn_on_unused_kwargs
from hydromt._utils.uris import strip_scheme
from hydromt.config import SETTINGS
from hydromt.data_adapter.caching import cache_vrt_tiles
from hydromt.drivers import RasterDatasetDriver
from hydromt.gis.merge import merge

logger: Logger = getLogger(__name__)


class RasterioDriver(RasterDatasetDriver):
    """Driver using rasterio for RasterDataset."""

    name = "rasterio"

    def read_data(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        time_range: Optional[TimeRange] = None,
        variables: Optional[Variables] = None,
        zoom_level: Optional[ZoomLevel] = None,
        metadata: Optional[SourceMetadata] = None,
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> xr.Dataset:
        """Read data using rasterio."""
        if metadata is None:
            metadata = SourceMetadata()
        # build up kwargs for open_raster
        warn_on_unused_kwargs(
            self.__class__.__name__,
            {"time_range": time_range, "zoom_level": zoom_level},
            logger=logger,
        )
        kwargs: Dict[str, Any] = {}

        # get source-specific options
        cache_root: str = str(
            self.options.get("cache_root", SETTINGS.cache_root),
        )

        if cache_root is not None and all([uri.endswith(".vrt") for uri in uris]):
            cache_dir = Path(cache_root) / self.options.get(
                "cache_dir",
                Path(
                    strip_scheme(uris[0])
                ).stem,  # default to first uri without extension
            )
            uris_cached = []
            for uri in uris:
                cached_uri: str = cache_vrt_tiles(
                    uri, geom=mask, cache_dir=cache_dir, logger=logger
                )
                uris_cached.append(cached_uri)
            uris = uris_cached

        # NoData part should be done in DataAdapter.
        if np.issubdtype(type(metadata.nodata), np.number):
            kwargs.update(nodata=metadata.nodata)
        # TODO: Implement zoom levels in https://github.com/Deltares/hydromt/issues/875
        # if zoom_level is not None and "{zoom_level}" not in uri:
        #     zls_dict, crs = self._get_zoom_levels_and_crs(uris[0], logger=logger)
        #     zoom_level = self._parse_zoom_level(
        #         zoom_level, mask, zls_dict, crs, logger=logger
        #     )
        #     if isinstance(zoom_level, int) and zoom_level > 0:
        #         # NOTE: overview levels start at zoom_level 1, see _get_zoom_levels_and_crs
        #         kwargs.update(overview_level=zoom_level - 1)
        ds = open_mfraster(uris, logger=logger, **kwargs)
        # rename ds with single band if single variable is requested
        if variables is not None and len(variables) == 1 and len(ds.data_vars) == 1:
            ds = ds.rename({list(ds.data_vars.keys())[0]: list(variables)[0]})
        return ds

    def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> None:
        """Write out a RasterDataset using rasterio."""
        pass


def open_raster(
    uri: Union[StrPath, IOBase, rasterio.DatasetReader, rasterio.vrt.WarpedVRT],
    mask_nodata: bool = False,
    chunks: Union[int, Tuple[int, ...], Dict[str, int], None] = None,
    nodata: Union[int, float, None] = None,
    logger: Logger = logger,
    **kwargs,
) -> xr.DataArray:
    """Open a gdal-readable file with rasterio based on.

    :py:meth:`rioxarray.open_rasterio`, but return squeezed DataArray.

    Arguments
    ---------
    filename : str, path, file-like, rasterio.DatasetReader, or rasterio.WarpedVRT
        Path to the file to open. Or already open rasterio dataset.
    mask_nodata : bool, optional
        set nodata values to np.nan (xarray default nodata value)
    nodata: int, float, optional
        Set nodata value if missing
    chunks : int, tuple or dict, optional
        Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
        ``{'x': 5, 'y': 5}``. If chunks is provided, it used to load the new
        DataArray into a dask array.
    **kwargs:
        key-word arguments are passed to :py:meth:`xarray.open_dataset` with
        "rasterio" engine.
    logger : logger object, optional
        The logger object used for logging messages. If not provided, the default
        logger will be used.

    Returns
    -------
    data : DataArray
        DataArray
    """
    chunks = chunks or {}
    kwargs.update(masked=mask_nodata, default_name="data", chunks=chunks)
    if not mask_nodata:  # if mask_and_scale by default True in xarray ?
        kwargs.update(mask_and_scale=False)
    if isinstance(uri, IOBase):  # file-like does not handle chunks
        logger.warning("Removing chunks to read and load remote data.")
        kwargs.pop("chunks")
    # keep only 2D DataArray
    da = rioxarray.open_rasterio(uri, **kwargs).squeeze(drop=True)
    # set missing _FillValue
    if mask_nodata:
        da.raster.set_nodata(np.nan)
    elif da.raster.nodata is None:
        if nodata is not None:
            da.raster.set_nodata(nodata)
        else:
            logger.warning(f"nodata value missing for {uri}")
    # there is no option for scaling but not masking ...
    scale_factor = da.attrs.pop("scale_factor", 1)
    add_offset = da.attrs.pop("add_offset", 0)
    if not mask_nodata and (scale_factor != 1 or add_offset != 0):
        raise NotImplementedError(
            "scale and offset in combination with mask_nodata==False is not supported."
        )
    return da


def open_mfraster(
    uris: Union[str, List[StrPath]],
    chunks: Union[int, Tuple[int, ...], Dict[str, int], None] = None,
    concat: bool = False,
    concat_dim: str = "dim0",
    mosaic: bool = False,
    mosaic_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> xr.Dataset:
    """Open multiple gdal-readable files as single Dataset with geospatial attributes.

    Each raster is turned into a DataArray with its name inferred from the filename.
    By default all DataArray are assumed to be on an identical grid and the output
    dataset is a merge of the rasters.
    If ``concat`` the DataArrays are concatenated along ``concat_dim`` returning a
    Dataset with a single 3D DataArray.
    If ``mosaic`` the DataArrays are concatenated along the the spatial dimensions
    using :py:meth:`~hydromt.raster.merge`.

    Arguments
    ---------
    uris: str, list of str/Path/file-like
        Paths to the rasterio/gdal files.
        Paths can be provided as list of paths or a path pattern string which is
        interpreted according to the rules used by the Unix shell. The variable name
        is derived from the basename minus extension in case a list of paths:
        ``<name>.<extension>`` and based on the file basename minus pre-, postfix and
        extension in a path pattern: ``<prefix><*name><postfix>.<extension>``
    chunks: int, tuple or dict, optional
        Chunk sizes along each dimension, e.g., 5, (5, 5) or {'x': 5, 'y': 5}.
        If chunks is provided, it used to load the new DataArray into a dask array.
    concat: bool, optional
        If True, concatenate raster along ``concat_dim``. We destinguish the following
        filenames from which the numerical index and variable name are inferred, where
        the variable name is based on the first raster.
        ``<name>_<index>.<extension>``
        ``<name>*<postfix>.<index>`` (PCRaster style; requires path pattern)
        ``<name><index>.<extension>``
        ``<name>.<extension>`` (index based on order)
    concat_dim: str, optional
        Dimension name of concatenate index, by default 'dim0'
    mosaic: bool, optional
        If True create mosaic of several rasters. The variable is named based on
        variable name infered from the first raster.
    mosaic_kwargs: dict, optional
        Mosaic key_word arguments to unify raster crs and/or resolution. See
        :py:meth:`hydromt.merge.merge` for options.
    **kwargs:
        key-word arguments are passed to :py:meth:`hydromt.raster.open_raster`

    Returns
    -------
    data : DataSet
        The newly created DataSet.
    """
    chunks = chunks or {}
    mosaic_kwargs = mosaic_kwargs or {}
    if concat and mosaic:
        raise ValueError("Only one of 'mosaic' or 'concat' can be True.")
    prefix, postfix = "", ""
    if isinstance(uris, str):
        if "*" in uris:
            prefix, postfix = basename(uris).split(".")[0].split("*")
        uris = [fn for fn in glob(uris) if not fn.endswith(".xml")]
    else:
        uris = [str(p) if isinstance(p, Path) else p for p in uris]
    if len(uris) == 0:
        raise OSError("no files to open")

    da_lst, index_lst, fn_attrs = [], [], []
    for i, uri in enumerate(uris):
        # read file
        da = open_raster(uri, chunks=chunks, **kwargs)

        # get name, attrs and index (if concat)
        if hasattr(uri, "path"):  # file-like
            bname = basename(uri.path)
        else:
            bname = basename(uri)
        if concat:
            # name based on basename until postfix or _
            vname = bname.split(".")[0].replace(postfix, "").split("_")[0]
            # index based on postfix behind "_"
            if "_" in bname and bname.split(".")[0].split("_")[1].isdigit():
                index = int(bname.split(".")[0].split("_")[1])
            # index based on file extension (PCRaster style)
            elif "." in bname and bname.split(".")[1].isdigit():
                index = int(bname.split(".")[1])
            # index based on postfix directly after prefix
            elif prefix != "" and bname.split(".")[0].strip(prefix).isdigit():
                index = int(bname.split(".")[0].strip(prefix))
            # index based on file order
            else:
                index = i
            index_lst.append(index)
        else:
            # name based on basename minus pre- & postfix
            vname = bname.split(".")[0].replace(prefix, "").replace(postfix, "")
            da.attrs.update(source_file=bname)
        fn_attrs.append(bname)
        da.name = vname

        if i > 0:
            if not mosaic:
                # check if transform, shape and crs are close
                if not da_lst[0].raster.identical_grid(da):
                    raise xr.MergeError("Geotransform and/or shape do not match")
                # copy coordinates from first raster
                da[da.raster.x_dim] = da_lst[0][da.raster.x_dim]
                da[da.raster.y_dim] = da_lst[0][da.raster.y_dim]
            if concat or mosaic:
                # copy name from first raster
                da.name = da_lst[0].name
        da_lst.append(da)

    if concat or mosaic:
        if concat:
            with dask.config.set(**{"array.slicing.split_large_chunks": False}):
                da = xr.concat(da_lst, dim=concat_dim)
                da.coords[concat_dim] = xr.IndexVariable(concat_dim, index_lst)
                da = da.sortby(concat_dim).transpose(concat_dim, ...)
                da.attrs.update(da_lst[0].attrs)
        else:
            da = merge(da_lst, **mosaic_kwargs)  # spatial merge
            da.attrs.update({"source_file": "; ".join(fn_attrs)})
        ds = da.to_dataset()  # dataset for consistency
    else:
        ds = xr.merge(
            da_lst
        )  # seems that with rioxarray drops all datarrays atrributes not just ds
        ds.attrs = {}

    # update spatial attributes
    if da_lst[0].rio.crs is not None:
        ds.rio.write_crs(da_lst[0].rio.crs, inplace=True)
    ds.rio.write_transform(inplace=True)
    return ds
