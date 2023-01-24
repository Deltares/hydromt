#!/usr/bin/env python
# -*- coding: utf-8 -*-
""""""
#%%
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import geopandas.array as geoarray
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from osgeo import osr
import pyproj
import shapely
import logging

from hydromt import gis_utils, raster
from hydromt.raster import XDIMS, YDIMS

from osgeo import __version__ as GDAL_VERSION

logger = logging.getLogger(__name__)

def Type(val):
    # try:
    #     s = eval(val)
    # except Exception:
    #     s = val
    return type(val)

def FieldType(lst):
    if float or np.float64 in lst:
        type = float
    else:
        type = int
    if str in lst:
        type = str
    return type

def DeterGeomType(wkt):
    from osgeo.ogr import CreateGeometryFromWkt
    geom_types = [CreateGeometryFromWkt(g).GetGeometryName() for g in wkt]

    if len(set(geom_types)) > 1:
        i = ["MULTI" in g for g in geom_types].index(True)
        geom_type = geom_types[i]
    else:
        geom_type = geom_types[0]
    del geom_types
    return geom_type

_linkTable = {
    int: "Integer64",
    float: "Real",
    str: "String",
}

class GeoBase(raster.XGeoBase):
    def __init__(self, xarray_obj):
        super(GeoBase, self).__init__(xarray_obj)
        self.set_meta()
        self._geom_dims = {
            "geometry": ("geometry",),
            "wkt": ("ogc_wkt",),
            "lonlat": (self.x_dim, self.y_dim),
            None: None,
        }

    @property
    def _all_names(self):
        names = [n for n in self._obj.coords]
        if isinstance(self._obj, xr.Dataset):
            names = names + [n for n in self._obj.data_vars]
        return names

    @property
    def _geom_names(self):
        names = []
        for name in self._all_names:
            if self._obj[name].ndim == 1 and isinstance(
                self._obj[name][0].values.item(), BaseGeometry
            ):
                names.append(name)
        return names

    def set_meta(self):
        """Determines the meta of the GeoDataset/ GeoDataArray
        Is invoked at Dataset/ DataArray creaton.

        INCLUDES
        --------
        object.x_dim
        object.y_dim
        object.index_dim
        """
        self.x_dim = None
        self.y_dim = None
        self.index_dim = list(self._obj.dims)[0]
        if any([item in XDIMS for item in self._all_names]):
            self.dtype = "lonlat"
            xind = [item in XDIMS for item in self._all_names].index(True)
            self.x_dim = self._all_names[xind]
            yind = [item in YDIMS for item in self._all_names].index(True)
            self.y_dim = self._all_names[yind]
            self.index_dim = list(self._obj[self.x_dim].dims)[0]
        elif "ogc_wkt" in self._all_names:
            self.dtype = "wkt" 
        elif "geometry" in self._all_names:
            self.dtype = "geometry"
        else:
            self.dtype = None

    @property
    def index(self):
        """Return the index values"""
        return self._obj[self.index_dim]

    @property
    def length(self):
        """Return the length of the index array"""
        return self._obj[self.index_dim].size

    @property
    def bounds(self):
        """Return the bounds (xmin, ymin, xmax, ymax) of the object."""
        return self.geometry.total_bounds

    @property
    def geometry(self) -> geoarray:
        """_summary_

        Returns
        -------
        geoarray
            Array containing geometries

        Raises
        ------
        ValueError
            Unknown geometry format
        """
        if self.dtype is None:
            raise ValueError("Unknown geometry format in Dataset")
        if self.dtype == "geometry":
            geoms = self._obj.geometry.values
            return geoarray.from_shapely(geoms, crs=self.crs)
        elif self.dtype == "lonlat":
            return geoarray.points_from_xy(
                self._obj[self.x_dim].values,
                self._obj[self.y_dim].values,
                crs=self.crs,
            )
        elif self.dtype == "wkt":
            return geoarray.from_wkt(
                self._obj.ogc_wkt.values,
                crs=self.crs,
            )

    # Internal conversion and selection methods
    # i.e. produces xarray.Dataset/ xarray.DataArray
    def ogr_compliant(self):
        """
        """

        wkt = [g.wkt for g in self.geometry]

        ## Determine Geometry type
        geom_type = DeterGeomType(wkt)

        ## Create the geometry DataArray
        ogc_wkt = xr.DataArray(
            data=wkt,
            dims="record",
            attrs={
                "long_name": "Geometry as ISO WKT",
                "grid_mapping": "spatial_ref",
            },
        )

        # Set spatial reference
        # srs = pyproj.CRS.from_epsg(self.crs.to_epsg())

        # crs = xr.DataArray(
        #     data=int(1),
        #     attrs={
        #         "long_name": "CRS definition",
        #         "crs_wkt": srs.to_wkt(),
        #         "spatial_ref": srs.to_wkt(),
        #     },
        # )

        return ogc_wkt, geom_type

    def to_crs(self, dst_crs):
        """Transform spatial coordinates to a new coordinate reference system.

        The ``crs`` attribute on the current GeoDataArray must be set.

        Arguments
        ----------
        dst_crs: int, dict, or str, optional
            Accepts EPSG codes (int or str); proj (str or dict) or wkt (str)

        Returns
        -------
        xr.DataArray
            DataArray with transformed geospatial coordinates
        """
        if self.crs is None:
            raise ValueError("Source CRS is missing. Use da.vector.set_crs(crs) first.")
        obj = self._obj.copy()
        geoms = self.geometry.to_crs(pyproj.CRS.from_user_input(dst_crs))
        
        obj = obj.drop_vars(self._geom_dims[self.dtype])

        obj = obj.assign_coords({"geometry": geoms})
        obj.vector.set_meta()
        obj.vector.set_crs(dst_crs)
        return obj

    def to_xy(self) -> xr.Dataset:
        """Converts Dataset/ DataArray to x,y structure

        Returns
        -------
        xr.Dataset
            x,y

        Raises
        ------
        ValueError
            Input data must be POINT geometries or POINT WKT-strings
        """

        geoms = self.geometry
        
        geom_type = DeterGeomType(
            [g.wkt for g in geoms]
        )

        if geom_type != "POINT":
            del geoms
            raise ValueError("Input data must be POINT geometries or POINT WKT-strings")

        obj = self._obj.copy()
        obj = obj.drop_vars(self._geom_dims[self.dtype])
        x,y = zip(*[(p.x,p.y) for p in geoms])
        del geoms
        obj = obj.assign_coords(
            {
                "x": (self.index_dim, list(x)),
                "y": (self.index_dim, list(y)),
            },
        )
        obj.vector.set_meta()
        return obj  

    def clip_geom(self, geom, predicate="intersects"):
        """Select all geometries that intersect with the input geometry.

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
        idx = gis_utils.filter_gdf(self.geometry, geom=geom, predicate=predicate)
        return self._obj.isel({self.index_dim: idx})

    def clip_bbox(self, bbox, crs=4326, buffer=None):
        """Select point locations to bounding box.

        Arguments
        ----------
        bbox: tuple of floats
            (xmin, ymin, xmax, ymax) bounding box
        buffer: float, optional
            buffer around bbox in crs units, None by default.

        Returns
        -------
        da: xarray.DataArray
            Clipped DataArray
        """
        if buffer is not None:
            bbox = np.atleast_1d(bbox)
            bbox[:2] -= buffer
            bbox[2:] += buffer
        idx = gis_utils.filter_gdf(self.to_gdf(), bbox=bbox, crs=crs, predicate="intersects")
        return self._obj.isel({self.index_dim: idx})

    # Constructers
    # i.e. from other datatypes or files
    def from_gdf():
        pass

    ## Output methods
    ## Either writes to files or other data types
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
        sdims = [self.y_dim, self.x_dim, self.index_dim, *self._geom_dims[self.dtype]]
        for name in self._all_names:
            dims = self._obj[name].dims
            if name not in sdims:
                # keep 1D variables with matching index_dim
                if len(dims) == 1 and dims[0] == self.index_dim:
                    gdf[name] = self._obj[name].values
                # keep reduced data variables
                elif reducer is not None and self.index_dim in self._obj[name].dims:
                    rdims = [
                        dim for dim in self._obj[name].dims if dim != self.index_dim
                    ]
                    gdf[name] = self._obj[name].reduce(reducer, rdim=dims)
        return gdf

    def to_netcdf(
        self,
        path: str,
        ogr_compliant: bool =  False,
        **kwargs,
    ):
        """Export geodataset vectordata to an ogr compliant netCDF4 file

        Parameters
        ----------
        root : str
            Directory in which the file is written to
        ogr_compliant : bool
            write the netCDF4 file in an ogr compliant format
            This makes it readable as a vector file in e.g. QGIS
        fname: : str
            Name of the file
        """

        if ogr_compliant:
            temp = self.ogr_compliant()

            temp.to_netcdf(path, engine="netcdf4", **kwargs)

            del temp
            return None
        
        temp = self._obj.copy()
        if self.dtype == "geometry":
            temp = temp.assign_coords(
                {
                    "ogc_wkt": [g.wkt for g in temp.geometry.values],
                },
            )
            temp = temp.drop_vars(["geometry"])
        temp.to_netcdf(path, engine="netcdf4", **kwargs)
        return None


@xr.register_dataarray_accessor("vector")
class GeoDataArray(GeoBase):
    def __init__(self, xarray_obj):
        super(GeoDataArray, self).__init__(xarray_obj)

    # Internal conversion and selection methods
    # i.e. produces xarray.Dataset/ xarray.DataArray
    def ogr_compliant(self, reducer=None) -> xr.Dataset:
        """Create a ogr compliant version of a xarray DataArray
        Note(!): The result will not be a DataArray

        Returns
        -------
        xr.Dataset
            ogr compliant
        """

        ogc_wkt, geom_type = super().ogr_compliant()
        
        out_ds = xr.Dataset()

        out_ds = out_ds.assign_coords(
            {
                "ogc_wkt": ogc_wkt,
            },
        )

        out_ds.vector.set_crs(self.crs.to_epsg())

        types = tuple(map(Type, self._obj.values))

        fld_type = FieldType(types)

        if self._obj.name is None: name = "value"
        else: name = self._obj.name

        values = None
        if not len(self._obj.dims) == 1 and list(self._obj.dims)[0] == self.index_dim:
            if reducer is not None:
                idd = self._obj.dims.index(self.index_dim)
                reduced_dims = list(range(len(self._obj.dims)))
                reduced_dims.remove(idd)
                values = reducer(self._obj.values, axis=tuple(reduced_dims))

        temp_da = xr.DataArray(
            data = self._obj.values,
            dims = "record",
            attrs={
                    "ogr_field_name": f"{name}",
                    "ogr_field_type": _linkTable[fld_type],
                },
            )

        if fld_type == str:
            temp_da.attrs.update({"ogr_field_width": 100})
        out_ds = out_ds.assign({f"{name}": temp_da})

        del temp_da

        out_ds = out_ds.assign_attrs(
            {
                "Conventions": "CF-1.6",
                "GDAL": f"GDAL {GDAL_VERSION}",
                "ogr_geometry_field": "ogc_wkt",
                "ogr_layer_type": f"{geom_type}",
            }
        )
        return out_ds

    # Constructers
    # i.e. from other datatypes or files
    @staticmethod
    def from_dataset(ds, crs=None) -> xr.Dataset:
        ds.vector.set_crs(crs)
        return ds

    @staticmethod
    def from_netcdf(path: str) -> xr.DataArray:
        """_summary_

        Parameters
        ----------
        path : str
            _description_

        Returns
        -------
        xr.DataArray
            _description_
        """

        temp = xr.open_dataarray(path)
        geoms = temp.vector.geometry

        da = xr.DataArray(
            data = temp.values,
            dims = "index",
            coords = {
                "geometry": ("index", geoms),
            },
            name = temp.name,
        )

        da.vector.set_crs(pyproj.CRS.from_wkt(temp.spatial_ref.crs_wkt))

        return da

    ## Output methods
    ## Either writes to files or other data types
    def to_gdf(self, reducer=None) -> gpd.GeoDataFrame:
        """Return geopandas GeoDataFrame with Point geometry based on DataArray
        coordinates. If a reducer is passed the DataArray variables are reduced along
        the all non-index dimensions and to a GeoDataFrame column.

        Arguments
        ---------
        reducer: callable
            input to ``xarray.DataArray.reducer`` func argument

        Returns
        -------
        gdf: gpd.GeoDataFrame
            GeoDataFrame
        """
        gdf = super().to_gdf(reducer)

        if self._obj.name is None: name = "value"
        else: name = self._obj.name

        gdf[name] = self._obj.values

        return gdf

@xr.register_dataset_accessor("vector")
class GeoDataset(GeoBase):
    def __init__(self, xarray_obj):
        super(GeoDataset, self).__init__(xarray_obj)

    # Internal conversion and selection methods
    # i.e. produces xarray.Dataset/ xarray.DataArray
    def ogr_compliant(self, reducer=None) -> xr.Dataset:
        """Create a ogr compliant version of a xarray Dataset

        Returns
        -------
        xr.Dataset
            ogr compliant
        """

        ogc_wkt, geom_type = super().ogr_compliant()

        out_ds = xr.Dataset()

        out_ds = out_ds.assign_coords(
            {"ogc_wkt": ogc_wkt},
            )

        # Set the spatial reference with the
        # set_crs form the GeoBase class
        out_ds.vector.set_crs(self.crs.to_epsg())

        for fld_header, da in self._obj.data_vars.items():
            values = None
            if not self.index_dim in da.dims:
                continue
            if not len(da.dims) == 1:                
                if reducer is not None:
                    idd = da.dims.index(self.index_dim)
                    reduced_dims = list(range(len(da.dims)))
                    reduced_dims.remove(idd)
                    values = reducer(da.values, axis=tuple(reduced_dims))
                else:
                    continue

            if values is None:
                values = da.values

            types = tuple(map(Type, values))

            fld_type = FieldType(types)

            temp_da = xr.DataArray(
                data=values,
                dims="record",
                attrs={
                    "ogr_field_name": f"{fld_header}",
                    "ogr_field_type": _linkTable[fld_type],
                },
            )

            if fld_type == str:
                temp_da.attrs.update({"ogr_field_width": 100})
            out_ds = out_ds.assign({f"{fld_header}": temp_da})

            del temp_da, values

        out_ds = out_ds.assign_attrs(
            {
                "Conventions": "CF-1.6",
                "GDAL": f"GDAL {GDAL_VERSION}",
                "ogr_geometry_field": "ogc_wkt",
                "ogr_layer_type": f"{geom_type}",
            }
        )
        return out_ds

    # Constructers
    # i.e. from other datatypes or filess
    @staticmethod
    def from_gdf(gdf: gpd.GeoDataFrame) -> xr.Dataset:
        """Creates Dataset with geospatial coordinates. The Dataset values are
        reindexed to the gdf index.

        Arguments
        ---------
        gdf: geopandas GeoDataFrame
            Spatial coordinates. The index should match the df index and the geometry
            columun may only contain Point geometries. Additional columns are also
            parsed to the xarray DataArray coordinates.

        Returns
        -------
        xr.Dataset
            Dataset with geospatial coordinates
        """
        if isinstance(gdf, gpd.GeoSeries):
            if gdf.name is None:
                gdf.name = "geometry"
            gdf = gdf.to_frame()
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError(f"gdf data type not understood {type(gdf)}")
        geom_name = gdf.geometry.name
        ds = gdf.to_xarray().set_coords(geom_name)
        ds.vector.set_crs(gdf.crs)
        return ds

    @staticmethod
    def from_dataset(ds, crs=None, geom_name=None, x_dim=None, y_dim=None) -> xr.Dataset:
        ds.vector.set_spatial_dims(geom_name=geom_name, x_dim=x_dim, y_dim=y_dim)
        ds.vector.set_crs(crs)
        return ds

    @staticmethod
    def from_netcdf(path: str) -> xr.Dataset:
        """Create GeoDataset from ogr compliant netCDF4 file

        Parameters
        ----------
        path : str
            Path to the netCDF4 file

        Returns
        -------
        xr.Dataset
            Dataset containing the geospatial data and attributes
        """
        temp = xr.open_dataset(path)
        geoms = temp.vector.geometry

        ds = xr.Dataset(
            coords={
                "index": temp.record.values,
                "geometry": ("index", geoms),
                "spatial_ref": temp.spatial_ref,
            }
        )

        for key, da in temp.drop_vars(
            [
                *temp.vector._geom_dims[temp.vector.dtype], 
                "spatial_ref",
            ]
            ).data_vars.items():
            temp_da = xr.DataArray(data=da.values, dims="index")
            ds = ds.assign({key: temp_da})

        ds.vector.set_crs(pyproj.CRS.from_wkt(temp.spatial_ref.crs_wkt))

        return ds

    def add_data(self, data_vars, coords=None, index_dim=None):
        """Align data along index axis and data to GeoDataset

        Arguments
        ---------
        data_vars: dict-like, DataArray or Dataset
            A mapping from variable names to `xarray.DataArray` objects.
            See :py:func:`xarray.Dataset` for all options.
            Additionally, it accepts `xarray.DataArray` with name property and `xarray.Dataset`.
        coords: sequence or dict of array_like, optional
            Coordinates (tick labels) to use for indexing along each dimension.
        index_dim: str, optional
            Name of index dimension in data_vars

        Returns
        -------
        ds: xarray.Dataset
            merged dataset
        """
        if isinstance(data_vars, xr.DataArray) and data_vars.name is not None:
            data_vars = data_vars.to_dataset()
        if isinstance(data_vars, xr.Dataset):
            ds_data = data_vars
        else:
            ds_data = xr.Dataset(data_vars, coords=coords)
        # check if any data array contain index_dim
        if self.index_dim not in ds_data.dims and index_dim in ds_data:
            ds_data = ds_data.rename({index_dim: self.index_dim})
        elif self.index_dim not in ds_data.dims:
            raise ValueError(f"Index dimension {self.index_dim} not found in dataset.")
        ds_data = ds_data.reindex({self.index_dim: self.index}).transpose(
            self.index_dim, ...
        )
        return xr.merge([self._obj, ds_data])
    
    ## Output methods
    ## Either writes to files or other data types
# %%
