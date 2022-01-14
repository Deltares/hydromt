#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convenience methods for reading and writing point  objects.
"""

import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
import pyproj
import logging

from . import gis_utils, raster

logger = logging.getLogger(__name__)


class GeoBase(raster.XGeoBase):
    """This is a vector extension for xarray.DataArray 1D time series data
    with geographical location information"""

    def __init__(self, xarray_obj):
        super(GeoBase, self).__init__(xarray_obj)
        # placeholder
        self._sindex = None

    @property
    def _all_names(self):
        names = [n for n in self._obj.coords]
        if isinstance(self._obj, xr.Dataset):
            names = names + [n for n in self._obj.data_vars]
        return names

    @property
    def x_dim(self):
        # NOTE overwrites raster.XGeoBase.x_dim
        if self.get_attrs("x_dim") not in self._all_names:
            self.set_spatial_dims()
        return self.attrs["x_dim"]

    @property
    def y_dim(self):
        # NOTE overwrites raster.XGeoBase.y_dim
        if self.get_attrs("y_dim") not in self._all_names:
            self.set_spatial_dims()
        return self.attrs["y_dim"]

    def set_spatial_dims(self, x_dim=None, y_dim=None, index_dim=None):
        """Set the spatial and index dimensions of the object.

        Arguments
        ----------
        x_dim, y_dim, index_dim: str, optional
            The name of the x, y and index dimensions.
        """
        # NOTE overwrites raster.XGeoBase.set_spatial_dims
        # infer x and index dims
        _obj = self._obj
        names = list(_obj.coords.keys())
        if isinstance(_obj, xr.Dataset):
            names = names + list(_obj.data_vars.keys())
        if x_dim is None:
            for dim in raster.XDIMS:
                if dim in names:
                    dim0 = index_dim if index_dim is not None else _obj[dim].dims[0]
                    if _obj[dim0].dims[0] == dim0:
                        x_dim = dim
                        index_dim = dim0
                        break
        if index_dim in _obj.dims and _obj[index_dim].dims[0] == index_dim:
            self.set_attrs(index_dim=index_dim)
        else:
            raise ValueError(
                "index dimension not found. Use 'set_spatial_dims'"
                + " functions with correct index_dim argument provided."
            )
        if x_dim in names and _obj[x_dim].dims[0] == index_dim:
            self.set_attrs(x_dim=x_dim)
        else:
            raise ValueError(
                "x dimension not found. Use 'set_spatial_dims'"
                + " functions with correct x_dim argument provided."
            )
        # infer y dim
        if y_dim is None:
            for dim in raster.YDIMS:
                if dim in names:
                    if _obj[dim].dims[0] == index_dim:
                        y_dim = dim
                        break
        # set dims
        if y_dim in names and _obj[y_dim].dims[0] == index_dim:
            self.set_attrs(y_dim=y_dim)
        else:
            raise ValueError(
                "y dimension not found. Use 'set_spatial_dims'"
                + " functions with correct y_dim argument provided."
            )

    @property
    def bounds(self):
        """Return the bounds (xmin, ymin, xmax, ymax) of the object."""
        xmin, xmax = self.xcoords.min(), self.xcoords.max()
        ymin, ymax = self.ycoords.min(), self.ycoords.max()
        return np.array([xmin, ymin, xmax, ymax])

    @property
    def total_bounds(self):
        """Return the bounds (xmin, ymin, xmax, ymax) of the object."""
        return self.bounds()  # mimic geopandas

    @property
    def index_dim(self):
        """Index dimension name."""
        if self.get_attrs("index_dim") not in self._obj.dims:
            self.set_spatial_dims()
        return self.attrs["index_dim"]

    @property
    def time_dim(self):
        """Time dimension name."""
        dim = self.get_attrs("time_dim")
        if dim not in self._obj.dims or np.dtype(self._obj[dim]).type != np.datetime64:
            self.set_attrs(time_dim=None)
            _dims = self._obj.dims
            tdims = [
                dim for dim in _dims if np.dtype(self._obj[dim]).type == np.datetime64
            ]
            if len(tdims) == 1:
                self.set_attrs(time_dim=tdims[0])
        return self.get_attrs("time_dim")

    @property
    def index(self):
        return self._obj[self.index_dim]

    @property
    def sindex(self):
        if self._sindex is None:
            # TODO find out if/ how this works at antimeridian line
            # TODO can we initiate this without geopandas in the middle?
            self._sindex = self.to_gdf().sindex
        return self._sindex

    @property
    def geometry(self):
        return gpd.points_from_xy(self.xcoords, self.ycoords)

    @property
    def has_sindex(self):
        """Check the existence of the spatial index without generating it.
        Use the `.sindex` attribute on a GeoDataFrame or GeoSeries
        to generate a spatial index if it does not yet exist,
        which may take considerable time based on the underlying index
        implementation.

        Note that the underlying spatial index may not be fully
        initialized until the first use.

        See Also
        ---------
        GeoDataFrame.has_sindex

        Returns
        -------
        bool
            `True` if the spatial index has been generated or
            `False` if not.
        """
        return self._sindex is not None

    def to_crs(self, dst_crs):
        """Transform spatial coordinates to a new coordinate reference system.

        The ``crs`` attribute on the current GeoDataArray must be set.

        Arguments
        ----------
        dst_crs: int, dict, or str, optional
            Accepts EPSG codes (int or str); proj (str or dict) or wkt (str)

        Returns
        -------
        da: xarray.DataArray
            DataArray with transformed geospatial coordinates
        """
        if self.crs is None:
            raise ValueError("Source CRS is missing. Use da.vector.set_crs(crs) first.")
        _obj = self._obj.copy()  # shallow
        gdf = self.to_gdf().to_crs(pyproj.CRS.from_user_input(dst_crs))
        # TODO rename depending on crs
        _obj.vector.set_crs(dst_crs)
        _obj[self.x_dim] = xr.IndexVariable(self.index_dim, gdf.geometry.x)
        _obj[self.y_dim] = xr.IndexVariable(self.index_dim, gdf.geometry.y)
        # reset spatial index
        self._sindex = None
        return _obj

    def clip_bbox(self, bbox, buffer=None, create_sindex=False):
        """Select point locations to bounding box.

        Arguments
        ----------
        bbox: tuple of floats
            (xmin, ymin, xmax, ymax) bounding box
        buffer: float, optional
            buffer around bbox in crs units, None by default.
        create_sindex: bool, optional
            Create spatial index to query the data if it does not yet exist, False by
            default.

        Returns
        -------
        da: xarray.DataArray
            Clipped DataArray
        """
        if buffer is not None:
            bbox = np.atleast_1d(bbox)
            bbox[:2] -= buffer
            bbox[2:] += buffer
        if not self.has_sindex and create_sindex == False:
            w, s, e, n = bbox
            idx = np.where(
                np.logical_and(
                    np.logical_and(self.xcoords >= w, self.xcoords <= e),
                    np.logical_and(self.ycoords >= s, self.ycoords <= n),
                )
            )[0]
        else:
            idx = self.sindex.intersection(bbox)
        return self._obj.isel({self.index_dim: idx})

    def clip_geom(self, geom, predicate="intersects"):
        """Select point locations to geometry. 


        Arguments
        ---------
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        predicate : {None, 'intersects', 'within', 'contains', \
                     'overlaps', 'crosses', 'touches'}, optional
            If predicate is provided, the input geometry is tested
            using the predicate function against each item in the
            index whose extent intersects the envelope of the input geometry:
            predicate(input_geometry, tree_geometry).
        
        Returns
        -------
        da: xarray.DataArray
            Clipped DataArray
        """
        idx = gis_utils.filter_gdf(self.to_gdf(), geom=geom, predicate=predicate)
        return self._obj.isel({self.index_dim: idx})


@xr.register_dataarray_accessor("vector")
class GeoDataArray(GeoBase):
    """This is a vector extension for xarray.DataArray 1D time series data
    with geographical location information"""

    @staticmethod
    def from_gdf(
        gdf,
        array_like,
        coords=None,
        dims=None,
        name=None,
        index_dim=None,
        keep_cols=True,
    ):
        """Parse GeoDataFrame object with point geometries to DataArray with
        geospatial attributes and merge with ``array_like`` data.

        Arguments
        ---------
        gdf: geopandas GeoDataFrame
            Spatial coordinates. The index should match the array_like index_dim and the
            geometry column may only contain Point geometries.
        array_like: array_like
            Values for this array. Must be an ``numpy.ndarray``, ndarray like,
            or castable to an ``ndarray``. If a self-described xarray or pandas
            object, attempts are made to use this array's metadata to fill in
            other unspecified arguments. A view of the array's data is used
            instead of a copy if possible.
        coords: sequence or dict of array_like, optional
            Coordinates (tick labels) to use for indexing along each dimension.
        dims: hashable or sequence of hashable, optional
            Name(s) of the data dimension(s). Must be either a hashable (only
            for 1D data) or a sequence of hashables with length equal to the
            number of dimensions. If this argument is omitted, dimension names
            default to ``['dim_0', ... 'dim_n']``.
        index_dim: str, optional
            Name of index dimension of array_like
        keep_cols: bool, optional
            If True, keep gdf columns as extra coordinates in dataset

        Returns
        -------
        da: xarray.DataArray
            DataArray with geospatial coordinates
        """
        # map gdf to ds coordinates
        scoords, index_dim = gdf_to_coords(gdf, index_dim, keep_cols)
        # add data
        da = xr.DataArray(array_like, dims=dims, coords=coords, name=name)
        # check if all data array contain index_dim
        if index_dim not in da.dims:
            raise ValueError(f"Index dimension {index_dim} not found on DataArray.")
        if np.dtype(da[index_dim]).type != np.dtype(gdf.index).type:
            try:
                da[index_dim] = da[index_dim].astype(np.dtype(gdf.index).type)
            except TypeError:
                raise TypeError(
                    "DataArray and GeoDataFrame index datatypes do not match."
                )
        da = da.reindex({index_dim: gdf.index}).assign_coords(scoords)
        # set geospatial attributes
        da.vector.set_spatial_dims(x_dim="x", y_dim="y", index_dim=index_dim)
        if gdf.crs is not None:
            da.vector.set_crs(gdf.crs)
        if da.dims[0] != index_dim:
            da = da.transpose(index_dim, ...)
        return da

    def to_gdf(self, reducer=None):
        """Return geopandas GeoDataFrame with Point geometry based on DataArray
        coordinates. If a reducer is passed the DataArray values are reduced along
        all non-index dimensions and saved in a column with same name as the GeoDataFrame.

        Arguments
        ---------
        reducer: callable
            input to ``xarray.DataArray.reducer`` func argument

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame
        """
        gdf = gpd.GeoDataFrame(index=self.index, geometry=self.geometry, crs=self.crs)
        gdf.index.name = self.index_dim
        # keep 1D variables with matching index_dim
        sdims = [self.y_dim, self.x_dim, self.index_dim, "geometry"]
        for name in self._obj.coords:
            dims = self._obj[name].dims
            if name not in sdims and len(dims) == 1 and dims[0] == self.index_dim:
                gdf[name] = self._obj[name].values
        # keep reduced data variable
        if reducer is not None:
            name = self._obj.name if self._obj.name is not None else "values"
            dims = [dim for dim in self._obj.dims if dim != self.index_dim]
            gdf[name] = self._obj.reduce(reducer, dim=dims)
        return gdf

    def interpolate(self, ds_like, method):
        # look at spatial index method for
        raise NotImplementedError()


@xr.register_dataset_accessor("vector")
class GeoDataset(GeoBase):
    """This is a vector extension for xarray.Dataset 1D time series data
    with geographical location information"""

    def __init__(self, xarray_obj):
        super(GeoDataset, self).__init__(xarray_obj)
        # placeholder
        self._sindex = None

    @staticmethod
    def from_gdf(gdf, data_vars={}, coords=None, index_dim=None, keep_cols=True):
        """Creates Dataset with geospatial coordinates. The Dataset values are
        reindexed to the gdf index.

        Arguments
        ---------
        gdf: geopandas GeoDataFrame
            Spatial coordinates. The index should match the df index and the geometry
            columun may only contain Point geometries. Additional columns are also
            parsed to the xarray DataArray coordinates.
        data_vars: dict-like, DataArray or Dataset
            A mapping from variable names to `xarray.DataArray` objects.
            See `xarray.Dataset` for all options.
            Aditionally it accepts `xarray.DataArray` with name property and `xarray.Dataset`.
        coords: sequence or dict of array_like, optional
            Coordinates (tick labels) to use for indexing along each dimension.
        index_dim: str, optional
            Name of index dimension in data_vars
        keep_cols: bool, optional
            If True, keep gdf columns as extra coordinates in dataset

        Returns
        -------
        da: xarray.Dataset
            Dataset with geospatial coordinates
        """
        # map gdf to ds coordinates
        scoords, index_dim = gdf_to_coords(gdf, index_dim, keep_cols)
        # add data
        if data_vars is not None and len(data_vars) > 0:
            if isinstance(data_vars, xr.DataArray) and data_vars.name is not None:
                data_vars = data_vars.to_dataset()
            if isinstance(data_vars, xr.Dataset):
                ds_data = data_vars
            else:
                ds_data = xr.Dataset(data_vars, coords=coords)
            # check if any data array contain index_dim
            if index_dim not in ds_data.dims:
                raise ValueError(f"Index dimension {index_dim} not found in dataset.")
            if np.dtype(ds_data[index_dim]).type != np.dtype(gdf.index).type:
                try:
                    ds_data[index_dim] = ds_data[index_dim].astype(
                        np.dtype(gdf.index).type
                    )
                except TypeError:
                    raise TypeError(
                        "Dataset and GeoDataFrame index datatypes do not match."
                    )
            ds = ds_data.reindex({index_dim: gdf.index}).assign_coords(scoords)
            ds = ds.transpose(index_dim, ...)
        else:
            ds = xr.Dataset(coords=scoords)
        # set geospatial attributes
        ds.vector.set_spatial_dims(x_dim="x", y_dim="y", index_dim=index_dim)
        if gdf.crs is not None:
            ds.vector.set_crs(gdf.crs)
        return ds

    @property
    def vars(self):
        """list: Returns non-coordinate varibles"""
        return list(self._obj.data_vars.keys())

    def to_gdf(self, reducer=None):
        """Return geopandas GeoDataFrame with Point geometry based on Dataset
        coordinates. If a reducer is passed the Dataset variables are reduced along
        the all non-index dimensions and to a GeoDataFrame column.

        Arguments
        ---------
        reducer: callable
            input to ``xarray.DataArray.reducer`` func argument

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame
        """
        gdf = gpd.GeoDataFrame(index=self.index, geometry=self.geometry, crs=self.crs)
        gdf.index.name = self.index_dim
        # keep 1D variables with matching index_dim
        sdims = [self.y_dim, self.x_dim, self.index_dim, "geometry"]
        for name in self._obj.reset_coords():
            dims = self._obj[name].dims
            if name not in sdims and len(dims) == 1 and dims[0] == self.index_dim:
                gdf[name] = self._obj[name].values
        # keep reduced data variables
        if reducer is not None:
            for name in self.vars:
                dims = [dim for dim in self._obj[name].dims if dim != self.index_dim]
                gdf[name] = self._obj[name].reduce(reducer, dim=dims)
        return gdf


def gdf_to_coords(gdf, index_dim=None, keep_cols=True):
    """Returns coordinate dictionary of gdf"""
    if not hasattr(gdf, "geometry"):
        raise ValueError("Unknown data type for gdf, provide geopandas object.")
    if not np.all(gdf.geometry.type == "Point"):
        raise ValueError("gdf may only contain Point geometry.")
    if index_dim is None:
        index_dim = gdf.index.name if gdf.index.name is not None else "index"
    coords = {
        index_dim: xr.IndexVariable(index_dim, gdf.index),
        "x": xr.IndexVariable(index_dim, gdf.geometry.x),
        "y": xr.IndexVariable(index_dim, gdf.geometry.y),
    }
    if keep_cols:
        for name in set(gdf.columns) - set(["geometry"]):
            coords.update({name: xr.Variable(index_dim, gdf[name])})
    return coords, index_dim
