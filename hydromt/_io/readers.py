"""Implementations for all of the necessary IO reading for HydroMT."""

import logging
from ast import literal_eval
from glob import glob
from io import IOBase
from os.path import abspath, basename, dirname, isfile, join, splitext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rioxarray
import xarray as xr
from pyogrio import read_dataframe
from pyproj import CRS
from requests import get as fetch
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import GEOMETRY_TYPES
from tomli import load as load_toml
from yaml import safe_load as load_yaml

from hydromt._typing.type_def import StrPath
from hydromt._utils.naming_convention import _expand_uri_placeholders, _placeholders
from hydromt._utils.path import _make_config_paths_abs
from hydromt._utils.uris import _is_valid_url
from hydromt.gis import _gis_utils, _vector_utils, raster, vector
from hydromt.gis.raster import GEO_MAP_COORD
from hydromt.gis.raster_merge import merge

if TYPE_CHECKING:
    from hydromt._validators.model_config import HydromtModelStep

logger = logging.getLogger(__name__)

__all__ = [
    "_open_mfcsv",
    "_open_raster",
    "_open_mfraster",
    "_open_raster_from_tindex",
    "_open_vector",
    "_open_geodataset",
    "_open_vector_from_table",
    "_open_timeseries_from_table",
    "_read_yaml",
    "_read_toml",
]

OPEN_VECTOR_PREDICATE = Literal[
    "intersects", "within", "contains", "overlaps", "crosses", "touches"
]
OPEN_VECTOR_DRIVER = Literal["csv", "xls", "xy", "vector", "parquet"]


def _open_mfcsv(
    paths: Dict[Union[str, int], Union[str, Path]],
    concat_dim: str,
    *,
    driver_kwargs: Optional[Dict[str, Any]] = None,
    variable_axis: Literal[0, 1] = 1,
    segmented_by: Literal["id", "var"] = "id",
) -> xr.Dataset:
    """Open multiple csv files as single Dataset.

    Arguments
    ---------
    paths : Dict[str | int, str | Path],
        Dictionary containing a id -> filename mapping. Here the ids,
        should correspond to the values of the `concat_dim` dimension and
        the corresponding setting of `segmented_by`. I.e. if files are
        segmented by id, these should contain ids. If the files are
        segmented by var, the keys of this dictionaires should be the
        names of the variables.
    concat_dim : str,
        name of the dimension that will be created by concatinating
        all of the supplied csv files.
    driver_kwargs : Dict[str, Any],
        Any additional arguments to be passed to pandas' `read_csv` function.
    variable_axis : Literal[0, 1] = 1,
        The axis along which your variables or ids are. so if the csvs have the
        columns as variable names, you would leave this as 1. If the variables
        are along the index, set this to 0. If you are unsure leave it as default.
    segmented_by: str
        How the csv files are segmented. Options are "id" or "var".  "id" should refer
        to the values of `concat_dim`. Segmented by id means csv files contain all
        variables for one id. Segmented by var or contain all ids for a
        single variable.

    Returns
    -------
    data : Dataset
        The newly created Dataset.
    """
    ds = xr.Dataset()
    if variable_axis not in [0, 1]:
        raise ValueError(f"there is no axis {variable_axis} available in 2D csv files")
    if segmented_by not in ["id", "var"]:
        raise ValueError(
            f"Unknown segmentation provided: {segmented_by}, options are ['var','id']"
        )

    csv_kwargs: Dict[str, Any] = {"index_col": 0}
    if driver_kwargs is not None:
        csv_kwargs.update(**driver_kwargs)

    # we'll just pick the first one we parse
    csv_index_name = None
    dfs = []
    for id, path in paths.items():
        df = pd.read_csv(path, **csv_kwargs)
        if variable_axis == 0:
            df = df.T

        if csv_index_name is None:
            # we're in the first loop
            if df.index.name is None:
                csv_index_name = "index"
            else:
                csv_index_name = df.index.name
        else:
            # could have done this in one giant boolean expression but throught
            # this was clearer
            if df.index.name is None:
                if not csv_index_name == "index":
                    logger.warning(
                        f"csv file {path} has inconsistent index name: {df.index.name}"
                        f"expected {csv_index_name} as it's the first one found."
                    )
            else:
                if not csv_index_name == df.index.name:
                    logger.warning(
                        f"csv file {path} has inconsistent index name: {df.index.name}"
                        f"expected {csv_index_name} as it's the first one found."
                    )

        if segmented_by == "id":
            df[concat_dim] = id
        elif segmented_by == "var":
            df["var"] = id
            df = df.reset_index().melt(id_vars=["var", "time"], var_name=concat_dim)
        else:
            raise RuntimeError(
                "Reached unknown segmentation branch (this should be impossible):"
                f" {segmented_by}, options are ['var','id']"
            )

        dfs.append(df)

    if segmented_by == "id":
        all_dfs_combined = (
            pd.concat(dfs, axis=0).reset_index().set_index([concat_dim, csv_index_name])
        )
    elif segmented_by == "var":
        all_dfs_combined = (
            pd.concat(dfs, axis=0)
            .pivot(index=[concat_dim, csv_index_name], columns="var")
            .droplevel(0, axis=1)
            .rename_axis(None, axis=1)
        )
    else:
        raise RuntimeError(
            "Reached unknown segmentation branch (this should be impossible):"
            f" {segmented_by}, options are ['var','id']"
        )
    ds = xr.Dataset.from_dataframe(all_dfs_combined)
    if "Unnamed: 0" in ds.data_vars:
        ds = ds.drop_vars("Unnamed: 0")
    return ds


def _open_raster(
    uri: Union[StrPath, IOBase, rasterio.DatasetReader, rasterio.vrt.WarpedVRT],
    *,
    mask_nodata: bool = False,
    chunks: Union[int, Tuple[int, ...], Dict[str, int], None] = None,
    nodata: Union[int, float, None] = None,
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
    da: xr.DataArray = rioxarray.open_rasterio(uri, **kwargs).squeeze(drop=True)
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


def _open_mfraster(
    uris: Union[str, List[StrPath]],
    *,
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
        uris = [path for path in glob(uris) if not path.endswith(".xml")]
    else:
        uris = [str(p) if isinstance(p, Path) else p for p in uris]
    if len(uris) == 0:
        raise OSError("no files to open")

    da_lst: List[xr.DataArray] = []
    index_lst, file_attrs = [], []
    for i, uri in enumerate(uris):
        # read file
        da = _open_raster(uri, chunks=chunks, **kwargs)

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
        file_attrs.append(bname)
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
            with dask.config.set(kwargs={"array.slicing.split_large_chunks": False}):
                da = xr.concat(da_lst, dim=concat_dim)
                da.coords[concat_dim] = xr.IndexVariable(concat_dim, index_lst)
                da = da.sortby(concat_dim).transpose(concat_dim, ...)
                da.attrs.update(da_lst[0].attrs)
        else:
            da = merge(da_lst, **mosaic_kwargs)  # spatial merge
            da.attrs.update({"source_file": "; ".join(file_attrs)})
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


def _open_raster_from_tindex(
    tindex_path,
    *,
    bbox=None,
    geom=None,
    tileindex="location",
    mosaic_kwargs=None,
    **kwargs,
):
    """Read and merge raster tiles.

    Raster tiles can potentially be in different CRS. Based on a
    tile index file as generated with `gdaltindex`. A bbox or geom describing the
    output area of interest is required.

    Arguments
    ---------
    tindex_path: path, str
        Path to tile index file.
    bbox : tuple of floats, optional
        (xmin, ymin, xmax, ymax) bounding box in EPGS:4326, by default None.
    geom : geopandas.GeoDataFrame/Series, optional
        A geometry defining the area of interest, by default None. The geom.crs
        defaults to EPSG:4326 if not set.
    tileindex: str
        Field name to hold the file path/location to the indexed rasters
    mosaic_kwargs: dict, optional
        Mosaic key_word arguments to unify raster crs and/or resolution. See
        :py:meth:`~hydromt.merge.merge()` for options.
    **kwargs:
        key-word arguments are passed to :py:meth:`hydromt.io.open_mfraster()`


    Returns
    -------
    data : Dataset
        A single-variable Dataset of merged raster tiles.
    """
    mosaic_kwargs = mosaic_kwargs or {}
    if bbox is not None and geom is None:
        geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
    if geom is None:
        raise ValueError("bbox or geom required in combination with tile_index")
    gdf = gpd.read_file(tindex_path)
    gdf = gdf.iloc[gdf.sindex.query(geom.to_crs(gdf.crs).union_all())]
    if gdf.index.size == 0:
        raise IOError("No intersecting tiles found.")
    elif tileindex not in gdf.columns:
        raise IOError(f'Tile index "{tileindex}" column missing in tile index file.')
    else:
        root = dirname(tindex_path)
        paths = []
        for path in gdf[tileindex]:
            path = Path(str(path))
            if not path.is_absolute():
                paths.append(Path(abspath(join(root, path))))
    # read & merge data
    if "dst_bounds" not in mosaic_kwargs:
        mosaic_kwargs.update(mask=geom)  # limit output domain to bbox/geom

    ds_out = _open_mfraster(
        paths, mosaic=len(paths) > 1, mosaic_kwargs=mosaic_kwargs, **kwargs
    )
    # clip to extent
    ds_out = ds_out.raster.clip_geom(geom)
    name = ".".join(basename(tindex_path).split(".")[:-1])
    ds_out = ds_out.rename({ds_out.raster.vars[0]: name})
    return ds_out  # dataset to be consitent with open_mfraster


def _open_geodataset(
    loc_path,
    *,
    data_path=None,
    var_name=None,
    index_dim=None,
    chunks=None,
    crs=None,
    bbox=None,
    geom=None,
    **kwargs,
) -> xr.Dataset:
    """Open and combine geometry location GIS file and timeseries file in a xr.Dataset.

    Arguments
    ---------
    loc_path: path, str
        Path to geometry location file, see :py:meth:`geopandas.read_file` for options.
        For point location, the file can also be a csv, parquet, xls(x) or xy file,
        see :py:meth:`hydromt.io.open_vector_from_table` for options.
    data_path: path, str
        Path to data file of which the index dimension which should match the geospatial
        coordinates index.
        This can either be a csv, or parquet with datetime in the first column and the
        location index in the header row, or a netcdf with a time and index dimensions.
        Note that this MUST be relative to the directory where `loc_path` points to
    var_name: str, optional
        Name of the variable in case of a csv, or parquet data_path file. By default,
        None and infered from basename.
    crs: str, `pyproj.CRS`, or dict
        Source coordinate reference system, ignored for files with a native crs.
    bbox : array of float, default None
        Filter features by given bounding box described by [xmin, ymin, xmax, ymax]
        Cannot be used with geom.
    index_dim:
        The dimension to index on.
    chunks:
        The dimensions of the chunks to store the underlying data in.
    geom : GeoDataFrame or GeoSeries | shapely Geometry, default None
        Filter for features that intersect with the geom.
        CRS mis-matches are resolved if given a GeoSeries or GeoDataFrame.
        Cannot be used with bbox.
    **kwargs:
        Key-word argument
    logger : logger object, optional
        The logger object used for logging messages. If not provided, the default
        logger will be used.

    Returns
    -------
    ds: xarray.Dataset
        Dataset with geospatial coordinates.
    """
    chunks = chunks or {}
    if not isfile(loc_path):
        raise IOError(f"GeoDataset point location file not found: {loc_path}")

    # For filetype [], only point geometry is supported
    filetype = str(loc_path).split(".")[-1].lower()
    if filetype in ["csv", "parquet", "xls", "xlsx", "xy"]:
        kwargs.update(assert_gtype="Point")
    # read geometry file
    polygon: Optional[Polygon] = box(*bbox) if bbox else None
    gdf = _open_vector(loc_path, crs=crs, bbox=polygon, geom=geom, **kwargs)
    if index_dim is None:
        index_dim = gdf.index.name if gdf.index.name is not None else "index"
    # read timeseries file
    if data_path is not None:
        loc_path_base = dirname(loc_path)
        full_data_path = join(loc_path_base, data_path)
        if isfile(full_data_path):
            da_ts = _open_timeseries_from_table(
                full_data_path, name=var_name, index_dim=index_dim
            )
            ds = vector.GeoDataset.from_gdf(gdf, da_ts)
        else:
            raise IOError(f"GeoDataset data file not found: {data_path}")
    else:
        ds = vector.GeoDataset.from_gdf(gdf)  # coordinates only
    return ds.chunk(chunks)


def _open_timeseries_from_table(path, *, name=None, index_dim="index", **kwargs):
    """Open timeseries csv or parquet file and parse to xarray.DataArray.

    Accepts files with time index on one dimension and numeric location index on the
    other dimension. In case of string location indices, non-numeric parts are
    filtered from the location index.

    Arguments
    ---------
    path: path, str
        Path to time series file
    name: str
        variable name, derived from basename of path if None.
    index_dim:
        the dimension to index on.
    **kwargs:
        key-word arguments are passed to the reader method

    Returns
    -------
    da: xarray.DataArray
        DataArray
    """
    _, ext = splitext(path)
    if ext == ".csv":
        csv_kwargs = dict(index_col=0, parse_dates=False)
        csv_kwargs.update(**kwargs)
        df = pd.read_csv(path, **csv_kwargs)
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path, **kwargs)
    else:
        raise ValueError(f"Unknown table file format: {ext}")

    first_index_elt = df.index[0]
    first_col_name = df.columns[0]

    try:
        if isinstance(first_index_elt, (int, float, np.number)):
            raise ValueError()
        pd.to_datetime(first_index_elt)
        # if this succeeds than axis 0 is the time dim
    except ValueError:
        try:
            if isinstance(first_col_name, (int, float, np.number)):
                raise ValueError()
            pd.to_datetime(first_col_name)
            df = df.T
        except ValueError:
            raise ValueError(f"No time index found in file: {path}")

    if np.dtype(df.index).type != np.datetime64:
        df.index = pd.to_datetime(df.index)

    # try parsing column index to integers
    if isinstance(df.columns[0], str):
        try:
            df.columns = [int("".join(filter(str.isdigit, n))) for n in df.columns]
            assert df.columns.size == np.unique(df.columns).size
        except (ValueError, AssertionError):
            raise ValueError(f"No numeric index found in file: {path}")
    df.columns.name = index_dim
    name = name if name is not None else basename(path).split(".")[0]
    return xr.DataArray(df, dims=("time", index_dim), name=name)


def _open_vector(
    path: StrPath,
    *,
    driver: Optional[OPEN_VECTOR_DRIVER] = None,
    crs: Optional[CRS] = None,
    dst_crs: Optional[CRS] = None,
    bbox: Optional[Tuple[float]] = None,
    geom: Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]] = None,
    assert_gtype: Optional[Union[Point, LineString, Polygon]] = None,
    predicate: OPEN_VECTOR_PREDICATE = "intersects",
    **kwargs,
):
    """Open fiona-compatible geometry, csv, parquet, excel or xy file and parse it.

    Construct a :py:meth:`geopandas.GeoDataFrame` CSV, parquet, or XLS file are
    converted to point geometries based on default columns names
    for the x- and y-coordinates, or if given, the x_dim and y_dim arguments.

    Parameters
    ----------
    path: str or Path-like,
        path to geometry file
    driver: {'csv', 'xls', 'xy', 'vector', 'parquet'}, optional
        driver used to read the file: :py:meth:`geopandas.open_file` for gdal vector
        files, :py:meth:`hydromt.io.open_vector_from_table`
        for csv, parquet, xls(x) and xy files. By default None, and inferred from
        file extension.
    crs: str, `pyproj.CRS`, or dict
        Source coordinate reference system, ignored for files with a native crs.
    dst_crs: str, `pyproj.CRS`, or dict
        Destination coordinate reference system.
    bbox : array of float, default None
        Filter features by given bounding box described by [xmin, ymin, xmax, ymax]
        Cannot be used with mask.
    geom : GeoDataFrame or GeoSeries | shapely Geometry, default None
        Filter for features that intersect with the mask.
        CRS mis-matches are resolved if given a GeoSeries or GeoDataFrame.
        Cannot be used with bbox.
    assert_gtype : {Point, LineString, Polygon}, optional
        If given, assert geometry type
    predicate : {'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches'},
        optional. If predicate is provided, the GeoDataFrame is filtered by testing
        the predicate function against each item. Requires bbox or mask.
        By default 'intersects'
    **kwargs:
        Keyword args to be passed to the driver method when opening the file

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Parsed geometry file
    """
    driver = driver if driver is not None else str(path).split(".")[-1].lower()
    if driver in ["csv", "parquet", "xls", "xlsx", "xy"]:
        gdf = _open_vector_from_table(path, driver=driver, **kwargs)
    # drivers with multiple relevant files cannot be opened directly, we should pass the uri only

    else:
        if driver == "pyogrio":
            if bbox:
                bbox_shapely = box(*bbox)
            else:
                bbox_shapely = None
            bbox_reader = _gis_utils._bbox_from_file_and_filters(
                str(path), bbox_shapely, geom, crs
            )
            gdf = read_dataframe(str(path), bbox=bbox_reader, **kwargs)
        else:
            if isinstance(bbox, list):
                bbox = tuple(bbox)
            gdf = gpd.read_file(str(path), bbox=bbox, mask=geom, **kwargs)

    # check geometry type
    if assert_gtype is not None:
        assert_gtype = np.atleast_1d(assert_gtype)
        if not np.all(np.isin(assert_gtype, GEOMETRY_TYPES)):
            gtype_err = assert_gtype[~np.isin(assert_gtype, GEOMETRY_TYPES)]
            raise ValueError(
                f"geometry type(s) {gtype_err} unknown, select from {GEOMETRY_TYPES}"
            )
        if not np.all(np.isin(gdf.geometry.type, assert_gtype)):
            raise ValueError(f"{path} contains other geometries than {assert_gtype}")

    # check if crs and filter
    if gdf.crs is None and crs is not None:
        gdf = gdf.set_crs(pyproj.CRS.from_user_input(crs))
    elif gdf.crs is None:
        raise ValueError("The GeoDataFrame has no CRS. Set one using the crs option.")
    if dst_crs is not None:
        gdf = gdf.to_crs(dst_crs)
    # filter points
    if gdf.index.size > 0 and (geom is not None or bbox is not None):
        idx = _vector_utils._filter_gdf(gdf, geom=geom, bbox=bbox, predicate=predicate)
        gdf = gdf.iloc[idx, :]
    return gdf


def _open_vector_from_table(
    path,
    *,
    driver=None,
    x_dim=None,
    y_dim=None,
    crs=None,
    **kwargs,
):
    r"""Read point geometry files from csv, parquet, xy or excel table files.

    Parameters
    ----------
    driver: {'csv', 'parquet', 'xls', 'xlsx', 'xy'}
        If 'csv' use :py:meth:`pandas.read_csv` to read the data;
        If 'parquet' use :py:meth:`pandas.read_parquet` to read the data;
        If 'xls' or 'xlsx' use :py:meth:`pandas.read_excel` with `engine=openpyxl`
        If 'xy' use :py:meth:`pandas.read_csv` with `index_col=False`, `header=None`,
        `sep='\s+'`.
    x_dim, y_dim: str
        Name of x, y column. By default the x-column header should be one of
        ['x', 'longitude', 'lon', 'long'], and y-column header one of
        ['y', 'latitude', 'lat']. For xy files, which don't have a header,
        the first column is interpreted as x and the second as y column.
    crs: int, dict, or str, optional
        Coordinate reference system, accepts EPSG codes (int or str), proj (str or dict)
        or wkt (str)
    path:
        The filename to read the table from.
    **kwargs
        Additional keyword arguments that are passed to the underlying drivers.

    Returns
    -------
    gdf: geopandas.GeoDataFrame
        Parsed and filtered point geometries
    """
    driver = driver.lower() if driver is not None else str(path).split(".")[-1].lower()
    if "index_col" not in kwargs and driver != "parquet":
        kwargs.update(index_col=0)
    if driver == "csv":
        df = pd.read_csv(path, **kwargs)
    elif driver == "parquet":
        df = pd.read_parquet(path, **kwargs)
    elif driver in ["xls", "xlsx"]:
        df = pd.read_excel(path, engine="openpyxl", **kwargs)
    elif driver == "xy":
        x_dim = x_dim if x_dim is not None else "x"
        y_dim = y_dim if y_dim is not None else "y"
        kwargs.update(index_col=False, header=None, sep=r"\s+")
        df = pd.read_csv(path, **kwargs).rename(columns={0: x_dim, 1: y_dim})
    else:
        raise IOError(f"Driver {driver} unknown.")
    # infer points from table
    df.columns = [c.lower() for c in df.columns]
    if x_dim is None:
        for dim in raster.XDIMS:
            if dim in df.columns:
                x_dim = dim
                break
    if x_dim is None or x_dim not in df.columns:
        raise ValueError(f'x dimension "{x_dim}" not found in columns: {df.columns}.')
    if y_dim is None:
        for dim in raster.YDIMS:
            if dim in df.columns:
                y_dim = dim
                break
    if y_dim is None or y_dim not in df.columns:
        raise ValueError(f'y dimension "{y_dim}" not found in columns: {df.columns}.')
    points = gpd.points_from_xy(df[x_dim], df[y_dim])
    gdf = gpd.GeoDataFrame(df.drop(columns=[x_dim, y_dim]), geometry=points, crs=crs)
    return gdf


def _read_workflow_yaml(
    path: StrPath,
) -> Tuple[str, Dict[str, Any], List["HydromtModelStep"]]:
    d = _read_yaml(path)
    modeltype = d.pop("modeltype", None)
    model_init = d.pop("global", {})

    # steps are required
    steps = d.pop("steps")

    return modeltype, model_init, steps


def _config_read(
    config_path: Union[Path, str],
    defaults: Optional[Dict[str, Any]] = None,
    abs_path: bool = False,
    skip_abspath_sections: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Read configuration/workflow file and parse to (nested) dictionary.

    Parameters
    ----------
    config_path : Union[Path, str]
        Path to configuration file
    defaults : dict, optional
        Nested dictionary with default options, by default dict()
    abs_path : bool, optional
        If True, parse string values to an absolute path if the a file or folder
        with that name (string value) relative to the config file exist,
        by default False
    skip_abspath_sections: list, optional
        These sections are not evaluated for absolute paths if abs_path=True,
        by default ['update_config']

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    defaults = defaults or {}
    skip_abspath_sections = skip_abspath_sections or ["setup_config"]
    # read
    ext = splitext(config_path)[-1].strip()
    if ext in [".yaml", ".yml"]:
        cfdict = _read_yaml(config_path)
    else:
        raise ValueError(f"Unknown extension: {ext} Hydromt only supports yaml")

    # parse absolute paths
    if abs_path:
        root = Path(dirname(config_path))
        cfdict = _make_config_paths_abs(cfdict, root, skip_abspath_sections)

    # update defaults
    if defaults:
        _cfdict = defaults.copy()
        _cfdict.update(cfdict)
        cfdict = _cfdict
    return cfdict


def _parse_values(
    cfdict: Dict[str, Any],
    skip_eval: bool = False,
    skip_eval_sections: Optional[List[str]] = None,
):
    """Parse string values to python default objects.

    Parameters
    ----------
    cfdict : dict
        Configuration dictionary.
    skip_eval : bool, optional
        Set true to skip evaluation, by default False
    skip_eval_sections : List, optional
        List of sections to skip evaluation, by default []

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary with evaluated values.
    """
    skip_eval_sections = skip_eval_sections or []
    # loop through two-level dict: section, key-value pairs
    for section in cfdict:
        # evaluate yaml items to parse to python default objects:
        if skip_eval or section in skip_eval_sections:
            cfdict[section].update(
                {key: str(var) for key, var in cfdict[section].items()}
            )  # cast None type values to str
            continue  # do not evaluate
        # numbers, tuples, lists, dicts, sets, booleans, and None
        for key, value in cfdict[section].items():
            try:
                value = literal_eval(value)
            except Exception:
                pass
            if isinstance(value, str) and len(value) == 0:
                value = None
            cfdict[section].update({key: value})
    return cfdict


def _read_nc(
    filename_template: StrPath,
    root: Path,
    *,
    mask_and_scale: bool = False,
    single_var_as_array: bool = True,
    load: bool = False,
    **kwargs,
) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
    """Read netcdf files at <root>/<file> and return as dict of xarray.Dataset.

    NOTE: Unless `single_var_as_array` is set to False a single-variable data source
    will be returned as :py:class:`xarray.DataArray` rather than
    :py:class:`xarray.Dataset`.
    key-word arguments are passed to :py:func:`xarray.open_dataset`.

    Parameters
    ----------
    file : str
        filename relative to model root, may contain wildcards
    mask_and_scale : bool, optional
        If True, replace array values equal to _FillValue with NA and scale values
        according to the formula original_values * scale_factor + add_offset, where
        _FillValue, scale_factor and add_offset are taken from variable attributes
        (if they exist).
    single_var_as_array : bool, optional
        If True, return a DataArray if the dataset consists of a single variable.
        If False, always return a Dataset. By default True.
    load : bool, optional
        If True, the data is loaded into memory. By default False.
    **kwargs:
        Additional keyword arguments that are passed to the `xr.open_dataset`
        function.

    Returns
    -------
    Dict[str, xr.Dataset]
        dict of xarray.Dataset
    """
    ncs = dict()
    path_template = root / filename_template

    path_glob, _, regex = _expand_uri_placeholders(
        str(path_template), placeholders=_placeholders
    )
    paths = glob(path_glob)
    for path in paths:
        name = ".".join(regex.match(path).groups())  # type: ignore
        # Load data to allow overwriting in r+ mode
        if load:
            ds: Union[xr.Dataset, xr.DataArray] = xr.open_dataset(
                path, mask_and_scale=mask_and_scale, **kwargs
            ).load()
            ds.close()
        else:
            ds = xr.open_dataset(path, mask_and_scale=mask_and_scale, **kwargs)
        # set geo coord if present as coordinate of dataset
        if GEO_MAP_COORD in ds.data_vars:
            ds = ds.set_coords(GEO_MAP_COORD)
        # single-variable Dataset to DataArray
        if single_var_as_array and len(ds.data_vars) == 1:
            (ds,) = ds.data_vars.values()
        ncs.update({name: ds})
    return ncs


def _read_yaml(path: StrPath) -> Dict[str, Any]:
    """Read yaml file and return as dict."""
    with open(path, "rb") as stream:
        yml = load_yaml(stream)

    return yml


def _parse_yaml(text: str) -> Dict[str, Any]:
    return load_yaml(text)


def _read_toml(path: StrPath) -> Dict[str, Any]:
    """Read toml file and return as dict."""
    with open(path, "rb") as f:
        data = load_toml(f)

    return data


def _yml_from_uri_or_path(uri_or_path: StrPath) -> Dict[str, Any]:
    if _is_valid_url(str(uri_or_path)):
        with fetch(str(uri_or_path), stream=True) as r:
            r.raise_for_status()
            yml = _parse_yaml(r.text)

    else:
        yml = _read_yaml(uri_or_path)
    return yml
